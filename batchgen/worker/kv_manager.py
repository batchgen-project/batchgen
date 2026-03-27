"""
KVCacheManager: GPU + host KV cache lifecycle management.

Extracted from batchgen_worker.py (Step 5 of scheduler split).
Handles: KV append to host, GPU KV allocation/release/destroy,
host KV utilization, watermark triggers.
"""
import logging
import os
import time
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.distributed as dist

from batchgen.worker.state import WorkerState
from batchgen.kv_cache.gpu_paged_kv_manager import GPUPagedKVCacheManager
from batchgen.kv_cache.host_kv_mananger_config import (
	build_gpu_kv_config,
	build_gpu_kv_config_aux,
)
from batchgen.kv_cache.dual_kv_cache_coordinator import DualKVCacheCoordinator
from batchgen.sequence import SequenceStatus

NUM_GPUS_PER_NODE = int(os.environ.get('NUM_GPUS_PER_NODE', '8'))
BATCHGEN_CB_DEBUG = os.environ.get("BATCHGEN_CB_LOG", "").upper() == "DEBUG"


class KVCacheManager:
	"""Manages GPU and host KV cache lifecycle.

	All shared state is accessed via WorkerState. Internal-only state
	(host_kv_page_stats, watermark counter) lives on this instance.
	"""

	def __init__(
		self,
		state: WorkerState,
		index_manager,  # IndexManager
		token_budget_fn: Optional[Callable[[int], int]] = None,
	):
		self.state = state
		self._index = index_manager
		self._token_budget_fn = token_budget_fn
		# Internal state (not shared)
		self._host_kv_page_stats: Optional[Dict] = None
		self._watermark_check_counter: int = 0

	# ---- Async KV offload ----

	def flush_deferred_kv(self) -> None:
		"""Flush all deferred KV host offload entries accumulated during forward.

		ONE event.synchronize() covers all layers, then batch-launch D2H copies.
		This replaces N per-layer syncs with a single post-forward sync.
		"""
		entries = self.state.deferred_kv_entries
		if not entries:
			return

		worker_view = getattr(self.state, 'deferred_kv_worker_view', None)
		batch_info = self.state.deferred_kv_batch
		if worker_view is None or batch_info is None:
			self.state.deferred_kv_entries = []
			return

		sequence_ids, sequence_lengths = batch_info

		# ONE sync for ALL layers — the key optimization
		if self.state.kv_offload_event is None:
			self.state.kv_offload_event = torch.cuda.Event()
		self.state.kv_offload_event.record(torch.cuda.current_stream(self.state.torch_device))
		self.state.kv_offload_event.synchronize()

		# Fire all D2H copies
		for layer_idx, k_tensor, v_tensor in entries:
			if k_tensor.dim() == 3:
				k_tensor = k_tensor.unsqueeze(2)
			if v_tensor is not None and v_tensor.dim() == 3:
				v_tensor = v_tensor.unsqueeze(2)

			task = worker_view.async_append_decode_kv_to_host(
				layer_idx=layer_idx,
				sequence_ids=sequence_ids,
				k_tensor=k_tensor,
				v_tensor=v_tensor,
				sequence_lengths=sequence_lengths,
			)

			self.state.pending_kv_tensors.append(k_tensor)
			if v_tensor is not None:
				self.state.pending_kv_tensors.append(v_tensor)
			if task is not None:
				self.state.pending_kv_tasks.append(task)

		# Throttle: prevent thread exhaustion from std::async
		if len(self.state.pending_kv_tasks) >= 256:
			self.wait_pending_tasks()

		self.state.deferred_kv_entries = []

	def append_kv_async(
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

		worker_view = getattr(self.state.core_engine, "host_paged_kv_worker_view", None)
		if worker_view is None:
			return

		sequence_ids = []
		sequence_lengths = []

		for local_idx in batch:
			uuid = self.state.local_to_uuid_map[local_idx]
			seq = self.state.global_batch.get_sequence(uuid)
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
					uuid = self.state.local_to_uuid_map.get(local_idx, "unknown")
					seq = self.state.global_batch.get_sequence(uuid) if uuid != "unknown" else None
					nan_seq_info.append({
						'batch_idx': idx,
						'local_idx': local_idx,
						'uuid': uuid[:8] if uuid != "unknown" else "unknown",
						'global_idx': seq.global_idx if seq else -1,
						'ctx_len': seq.current_context_length if seq else -1,
					})
			logging.error(
				f"[KV-NaN-DETECT] Rank {self.state.rank}: NaN detected in k_tensor BEFORE host append! "
				f"layer={layer_idx}, k_tensor_shape={list(k_tensor.shape)}, "
				f"affected_seqs={nan_seq_info}"
			)

		# Ensure compute stream finishes writing k_tensor before D2H copy reads it.
		# Record event on compute stream, then event.synchronize() waits only for
		# work up to this point (not all GPU work). Lighter than full device sync.
		if self.state.kv_offload_event is None:
			self.state.kv_offload_event = torch.cuda.Event()
		self.state.kv_offload_event.record(torch.cuda.current_stream(self.state.torch_device))
		self.state.kv_offload_event.synchronize()

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
		self.state.pending_kv_tensors.append(k_tensor)
		if v_tensor is not None:
			self.state.pending_kv_tensors.append(v_tensor)

		# Add to pending list - will be waited at page boundary
		self.state.pending_kv_tasks.append(task)

		# THROTTLING FIX: Prevent "Resource temporarily unavailable" (EAGAIN) error
		# std::async creates a new thread for each task. With 61 layers and 64 tokens
		# per boundary, we can hit ~3900 concurrent threads per boundary interval.
		# Wait and clear when threshold is reached to avoid exhausting system thread limits.
		# Threshold: 256 tasks (conservative to leave room for other threads)
		MAX_PENDING_KV_TASKS = 256
		if len(self.state.pending_kv_tasks) >= MAX_PENDING_KV_TASKS:
			self.wait_pending_tasks()

	def wait_pending_tasks(self) -> int:
		"""
		Wait for all pending KV append tasks at page boundary.
		Returns the number of tasks that were waited for.

		CRITICAL: Also syncs CUDA to ensure all D2H DMA operations complete.
		Without this, KV data may not be fully written to host memory when
		sequences are later resumed, causing KV corruption.
		"""
		num_tasks = len(self.state.pending_kv_tasks)
		for task in self.state.pending_kv_tasks:
			if task is not None:
				task.wait()

		# CRITICAL FIX: Sync CUDA after waiting for tasks
		# The async tasks use a separate CUDA stream for D2H copies.
		# Even though each task internally syncs its stream via cudaEventSynchronize,
		# we need a full device sync to ensure ALL pending operations complete
		# before we allow GPU pages to be freed/reused.
		if num_tasks > 0:
			torch.cuda.synchronize(self.state.torch_device)

		self.state.pending_kv_tasks.clear()

		# CRITICAL: Clear tensor references AFTER tasks complete
		# Tensors can now be safely garbage collected / memory reused
		self.state.pending_kv_tensors.clear()

		return num_tasks

	# ---- GPU KV manager lifecycle ----

	def compute_sequence_tokens(self, sequence_ids: List[int]) -> List[int]:
		"""Reuse cached token budgets so host/GPU allocations stay consistent."""
		return [self._token_budget_fn(sequence_id) for sequence_id in sequence_ids]

	def bind_gpu_manager(self, manager) -> None:
		"""Bind GPU KV manager to both worker state and core_engine.

		If manager is a DualKVCacheCoordinator, the primary manager is bound
		to existing gpu_paged_kv_manager slots and the auxiliary (indexer) is
		bound to gpu_paged_kv_manager_aux slots.
		"""
		self.state.gpu_kv_manager = manager
		if isinstance(manager, DualKVCacheCoordinator):
			if hasattr(self.state.core_engine, "gpu_paged_kv_manager"):
				self.state.core_engine.gpu_paged_kv_manager = manager.primary
			if hasattr(self.state.core_engine, "gpu_paged_kv_manager_aux"):
				self.state.core_engine.gpu_paged_kv_manager_aux = manager.auxiliary
		else:
			if hasattr(self.state.core_engine, "gpu_paged_kv_manager"):
				self.state.core_engine.gpu_paged_kv_manager = manager

	def ensure_gpu_manager(self, sequence_tokens: Sequence[int]) -> GPUPagedKVCacheManager:
		"""Return a GPU paged KV manager with enough pages for `sequence_tokens`.

		For DSA models, returns a DualKVCacheCoordinator wrapping both primary
		(MLA) and auxiliary (indexer) managers.
		"""
		gpu_config = build_gpu_kv_config(
			model_name=self.state.huggingface_ckpt_name,
			sequence_tokens=sequence_tokens,
		)

		manager = self.state.gpu_kv_manager
		required_pages = gpu_config.num_pages
		current_pages = (
			getattr(getattr(manager, "config", None), "num_pages", 0)
			if manager is not None
			else 0
		)

		if manager is not None and current_pages >= required_pages:
			manager.initialize()
			self.bind_gpu_manager(manager)
			return manager

		if manager is not None:
			manager.destroy()

		logging.info(
			"Rank %s creating GPUPagedKVCacheManager on %s: "
			"current pages=%d, required pages=%d",
			self.state.rank, self.state.local_rank, current_pages, required_pages
		)

		primary = GPUPagedKVCacheManager(
			config=gpu_config,
			device=self.state.local_rank,
		)

		# For DSA models, create auxiliary (indexer) manager and wrap in coordinator
		aux_config = build_gpu_kv_config_aux(
			model_name=self.state.huggingface_ckpt_name,
			sequence_tokens=sequence_tokens,
		)
		if aux_config is not None:
			auxiliary = GPUPagedKVCacheManager(
				config=aux_config,
				device=self.state.local_rank,
			)
			manager = DualKVCacheCoordinator(primary, auxiliary)
			manager.initialize()
			self.bind_gpu_manager(manager)

			logging.info(
				"Rank %s initialized DualKVCacheCoordinator on %s: "
				"primary=%d pages (dim=%d), auxiliary=%d pages (dim=%d)",
				self.state.rank, self.state.local_rank,
				gpu_config.num_pages, gpu_config.k_head_dim,
				aux_config.num_pages, aux_config.k_head_dim,
			)
		else:
			manager = primary
			manager.initialize()
			self.bind_gpu_manager(manager)

			logging.info(
				"Rank %s initialized GPUPagedKVCacheManager on %s with %d pages",
				self.state.rank, self.state.local_rank, gpu_config.num_pages,
			)
		return manager

	def prepare_gpu_kv(self, local_sequence_ids: List[int]) -> None:
		"""Allocate GPU KV pages and load host-resident KV for the batch."""
		if not local_sequence_ids:
			return

		# Convert local indices to global_idx (consistent with host KV registration)
		global_sequence_ids = self._index.local_indices_to_global_seq_ids(local_sequence_ids)

		sequence_tokens = self.compute_sequence_tokens(local_sequence_ids)
		manager = self.ensure_gpu_manager(sequence_tokens)

		logging.info(
			f"Rank {self.state.rank} Allocating GPU KV pages for global_idx: {global_sequence_ids}"
		)

		# allocate_pages_for_sequences implicitly registers the sequences
		manager.allocate_pages_for_sequences(global_sequence_ids, sequence_tokens)
		manager.rebuild_page_table(global_sequence_ids)
		self.load_host_to_gpu(manager, global_sequence_ids)

	def load_host_to_gpu(
		self,
		manager: GPUPagedKVCacheManager,
		global_sequence_ids: List[int],
	) -> None:
		"""Copy prefetched host KV pages into the GPU cache."""
		if not global_sequence_ids:
			return
		copy_start = time.perf_counter()
		worker_view = getattr(self.state.core_engine, "host_paged_kv_worker_view", None)
		if worker_view is None:
			raise RuntimeError("Host paged KV worker view is not bound to the core engine")

		# DIAGNOSTIC: Check if these are resuming sequences (have decoded tokens)
		resuming_seq_info = []
		for global_idx in global_sequence_ids:
			# Find the sequence by global_idx
			for uuid, local_idx in self.state.uuid_to_local_map.items():
				seq = self.state.global_batch.get_sequence(uuid)
				if seq and seq.global_idx == global_idx and seq.decoded_length > 0:
					resuming_seq_info.append({
						'global_idx': global_idx,
						'decoded_length': seq.decoded_length,
						'current_context_length': seq.current_context_length,
					})
					break

		if resuming_seq_info and BATCHGEN_CB_DEBUG:
			logging.debug(
				f"Rank {self.state.rank}: load_host_to_gpu loading KV for {len(resuming_seq_info)} RESUMING sequences. First 5: {resuming_seq_info[:5]}"
			)

		sequence_tensor = torch.tensor(global_sequence_ids, dtype=torch.int64, device="cpu")
		k_ptrs, v_ptrs = manager.get_padded_3d_page_pointers()
		active_sequence_page_counts = manager.export_active_sequence_page_counts()

		logging.debug(
			f"Rank {self.state.rank}: load_host_to_gpu launching async load for "
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
		torch.cuda.synchronize(self.state.torch_device)

		load_duration = time.perf_counter() - copy_start
		logging.debug(
			"Rank %s Loaded host KV for %d sequences into GPU cache in %.3fs",
			self.state.rank, len(global_sequence_ids), load_duration,
		)

	def release_gpu_pages(self, local_sequence_ids: List[int]) -> None:
		"""Return GPU KV pages associated with the provided local sequence ids."""
		manager = self.state.gpu_kv_manager
		if manager is None or not local_sequence_ids:
			return

		global_sequence_ids = self._index.local_indices_to_global_seq_ids(local_sequence_ids)

		if not global_sequence_ids:
			return

		try:
			manager.free_pages_for_sequences(global_sequence_ids)
			# NOTE: No sync needed - page deallocation is synchronous to the allocator
			logging.debug(
				f"Rank {self.state.rank} Released GPU KV pages for global_idx: {global_sequence_ids}"
			)

			# FIX Bug 2: Remove from tracking set and reset gpu_pages_allocated
			for local_idx in local_sequence_ids:
				uuid = self.state.local_to_uuid_map.get(local_idx)
				if uuid:
					self.state.sequences_with_gpu_kv.discard(uuid)
					seq = self.state.global_batch.get_sequence(uuid)
					if seq is not None:
						seq.gpu_pages_allocated = 0

		except KeyError as exc:
			logging.warning(
				"Rank %s failed to release GPU KV pages for %s: %s",
				self.state.rank, global_sequence_ids, exc,
			)

	def destroy_gpu_kv(self, *, empty_cuda_cache: bool = False) -> None:
		"""Destroy the GPU paged KV cache manager if it is present."""
		manager = self.state.gpu_kv_manager
		if manager is None:
			return

		# DIAGNOSTIC: Log state before destruction for KV corruption investigation
		if self.state.global_batch is not None:
			seqs_with_gpu_alloc = []
			for seq in self.state.global_batch:
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
					f"Rank {self.state.rank}: destroy_gpu_kv called with "
					f"{len(seqs_with_gpu_alloc)} sequences having GPU allocation state. "
					f"First 5: {seqs_with_gpu_alloc[:5]}"
				)

		manager.destroy(empty_cuda_cache=empty_cuda_cache)

		# FIX Bug 2: Clear tracking set when GPU KV is destroyed
		self.state.sequences_with_gpu_kv.clear()

		# CRITICAL FIX: Reset GPU allocation state for ALL non-completed sequences
		# Without this, sequences retain stale had_initial_gpu_reservation=True,
		# causing them to get insufficient GPU buffer on resume after prefill interruption
		if self.state.global_batch is not None:
			reset_count = 0
			for seq in self.state.global_batch:
				if seq.status != SequenceStatus.COMPLETED:
					if seq.gpu_pages_allocated > 0 or seq.had_initial_gpu_reservation:
						if BATCHGEN_CB_DEBUG:
							logging.debug(
								f"Rank {self.state.rank}: Resetting GPU state for {seq.uuid[:8]} "
								f"(status={seq.status.name}, gpu_pages={seq.gpu_pages_allocated}, "
								f"had_initial={seq.had_initial_gpu_reservation})"
							)
						seq.reset_gpu_allocation()
						reset_count += 1
			if reset_count > 0:
				logging.info(
					f"Rank {self.state.rank}: Reset GPU allocation state for {reset_count} sequences"
				)

	# ---- Queries ----

	def get_host_free_pages(self) -> int:
		"""Get current free pages from host KV cache."""
		stats = self.state.host_kv_view.get_stats()
		return stats.num_free_pages

	def get_gpu_free_pages(self) -> int:
		"""Get current free pages from GPU KV cache."""
		manager = self.state.gpu_kv_manager
		if manager is None:
			return 0
		return manager.get_stats().num_free_pages

	def get_host_utilization(self) -> Dict[str, int]:
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
		stats = self.state.host_kv_view.get_stats()

		# Count pages used by sequences with KV in host on THIS NODE (all ranks on node)
		# Host KV is shared across all GPUs on a node
		node_id = self.state.rank // NUM_GPUS_PER_NODE
		node_rank_start = node_id * NUM_GPUS_PER_NODE
		node_rank_end = min(node_rank_start + NUM_GPUS_PER_NODE, self.state.world_size)

		# CRITICAL FIX: IN_DECODE sequences also have KV in host (streams after each layer)
		valid_statuses = {SequenceStatus.PREFILLED, SequenceStatus.ON_HOLD, SequenceStatus.IN_DECODE}

		# Count sequences per status for detailed logging
		status_counts = {status: [] for status in valid_statuses}
		for rank_on_node in range(node_rank_start, node_rank_end):
			for status in valid_statuses:
				seqs = self.state.global_batch.get_sequences_for_rank_with_status(rank_on_node, status)
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

		if self.state.local_rank == 0:
			logging.debug(
				f"[HOST_KV_UTIL] C++ stats: used={used_pages}, free={free_pages}, "
				f"total={stats.num_total_pages}, {len(valid_sequences)} valid seqs"
			)

		return {
			'rank': self.state.rank,
			'node_id': self.state.rank // NUM_GPUS_PER_NODE,
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

	def check_watermark_trigger(self) -> bool:
		"""Check if any node exceeds host KV free page watermark.

		Watermark = 70% FREE (underutilized).
		Only checks if this rank is local_rank 0 (one check per node).

		Returns:
			True if should interrupt decode and switch to prefill
		"""
		if not self.state.enable_decode_preemption:
			return False

		# Only local_rank 0 reports (one per node)
		if self.state.local_rank == 0:
			local_stats = self.get_host_utilization()
		else:
			local_stats = None

		# Gather stats from all local_rank 0 representatives
		all_stats = [None] * self.state.world_size
		dist.all_gather_object(all_stats, local_stats)

		# Filter to only node representatives
		node_stats = [s for s in all_stats if s is not None]

		if not node_stats:
			return False

		# Check if any node above watermark (too much free space)
		max_free_percent = max(s['free_percent'] for s in node_stats)
		above_watermark = max_free_percent > self.state.host_kv_watermark

		# Check if queued or evicted sequences available
		has_queued = self.state.global_batch.has_queueing()
		has_evicted = self.state.enable_host_kv_eviction and self.state.global_batch.has_evicted()

		should_trigger = above_watermark and (has_queued or has_evicted)

		# Log global host KV cache stats (rank 0 only, aggregated across all nodes)
		if self.state.rank == 0:
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
					f"[Host KV Cache] PREFILL TRIGGER: max_node_free={max_free_percent}% > {self.state.host_kv_watermark}%, "
					f"queued_sequences={len(self.state.global_batch.get_sequences_by_status(SequenceStatus.QUEUEING))}"
				)
				for s in node_stats:
					logging.info(
						f"[Host KV Cache]   Node {s['node_id']}: {s['num_used_pages']}/{s['num_total_pages']} "
						f"pages ({100-s['free_percent']}% used, {s['free_percent']}% free)"
					)
		else:
			# Log summary even when not triggering (every 10th check to avoid spam)
			self._watermark_check_counter += 1
			if self._watermark_check_counter % 10 == 0:
				logging.debug(
					f"[Host KV Cache] Check #{self._watermark_check_counter}: max_free={max_free_percent}%, "
					f"threshold={self.state.host_kv_watermark}%, has_queued={has_queued}, trigger={should_trigger}"
				)

		return should_trigger
