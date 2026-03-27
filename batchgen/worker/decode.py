"""
DecodeScheduler: Batch selection, model loading, configuration, and continuous decode loop.

Extracted from batchgen_worker.py (Step 9 of scheduler split).
Handles: decode batch selection, model loading, GPU KV allocation, continuous decode loop.
"""
import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from batchgen.worker.state import WorkerState
from batchgen.kv_cache.gpu_paged_kv_manager import GPUPagedKVCacheManager
from batchgen.kv_cache.dual_kv_cache_coordinator import DualKVCacheCoordinator
from batchgen.continuous_batching import FastBoundaryTimingStats
from batchgen.models.wrappers import AttnWrapperBase
from batchgen.sequence import SequenceStatus

# Attn_Wrapper is an alias for AttnWrapperBase
Attn_Wrapper = AttnWrapperBase

BATCHGEN_CB_DEBUG = os.environ.get("BATCHGEN_CB_LOG", "").upper() == "DEBUG"
BATCHGEN_SYNC_KV = os.environ.get("BATCHGEN_SYNC_KV", "0") == "1"


class DecodeScheduler:
	"""Decode batch selection, model loading, and continuous decode loop.

	Uses a worker reference for shared methods and the boundary handler.
	"""

	def __init__(self, state: WorkerState, index_manager, kv_manager, boundary_handler, worker):
		self.state = state
		self._index = index_manager
		self._kv = kv_manager
		self._boundary = boundary_handler
		self._worker = worker

	def prepare_batch(self) -> List[str]:
		"""
		Select sequences for decode phase from PREFILLED sequences.
		Greedily fill GPU KV cache to ~90% capacity.
		"""
		prefilled_uuids = self.state.global_batch.get_sequences_by_status(SequenceStatus.PREFILLED)
		onhold_uuids = self.state.global_batch.get_sequences_by_status(SequenceStatus.ON_HOLD)
		
		# Combine and sort for deterministic ordering
		all_candidates = prefilled_uuids + onhold_uuids
		all_candidates.sort(key=lambda uuid: self.state.global_batch.get_sequence(uuid).global_idx)
		
  
		if not all_candidates:
			return []
		
		# Get GPU page capacity - GPU KV manager must be initialized before batch selection
		# (model loading and GPU KV init happen in generate() BEFORE this call)
		if self._worker.gpu_paged_kv_cache_manager is None or not self._worker.gpu_paged_kv_cache_manager.is_initialized:
			raise RuntimeError(
				"GPU KV manager must be initialized before _prepare_decode_batch(). "
				"Ensure _load_decode_model() and _init_gpu_kv_with_actual_size() are called first."
			)
		total_pages = self._worker.gpu_paged_kv_cache_manager.get_stats().num_total_pages
		
		# 90% watermark
		capacity_per_rank = int(total_pages * 0.9)
		
		# Greedily fill
		rank_pages_used = [0] * self.state.world_size
		decode_batch = []
		
		for uuid in all_candidates:
			seq = self.state.global_batch.get_sequence(uuid)
			assigned_rank = seq.assigned_rank
			req_pages = seq.get_gpu_pages_for_two_page_buffer()
			
			if rank_pages_used[assigned_rank] + req_pages <= capacity_per_rank:
				decode_batch.append(uuid)
				rank_pages_used[assigned_rank] += req_pages

		if self.state.rank == 0:
			logging.info(
				f"[DECODE] Prepared batch: {len(decode_batch)} sequences"
			)

		return decode_batch

	def try_load_new(
		self, 
		current_decode_uuids: List[str],
		current_local_indices: List[int]
	) -> Tuple[List[str], List[int]]:
		"""
		Load PREFILLED sequences from Host KV to GPU KV if space available.
		Maintains deterministic ordering across all ranks.
		"""
		gpu_free_pages = self._kv.get_gpu_free_pages()
		candidates = self.state.global_batch.get_sequences_by_status(SequenceStatus.PREFILLED)
		
		# Sort for deterministic ordering across all ranks
		candidates.sort(key=lambda u: self.state.global_batch.get_sequence(u).global_idx)
		
		new_uuids = []
		pages_needed = 0
		
		for uuid in candidates:
			seq = self.state.global_batch.get_sequence(uuid)
			req = seq.get_pages_required()
			if pages_needed + req <= gpu_free_pages:
				new_uuids.append(uuid)
				pages_needed += req
			else:
				break
		
		if not new_uuids:
			return current_decode_uuids, current_local_indices

		# Get local indices for sequences belonging to THIS rank
		new_local_indices = self._index.get_local_indices_for_uuids(new_uuids)
		
		if new_local_indices:
			# Allocate and load (without final rebuild)
			self._worker._allocate_and_load_gpu_kv_for_new_sequences(new_local_indices)

		# Update status AFTER load completes
		self._worker._update_batch_status(new_uuids, SequenceStatus.IN_DECODE)
		
		# Build updated lists
		updated_uuids = current_decode_uuids + new_uuids
		updated_batch = current_local_indices + new_local_indices
		
		# Final page table rebuild with ALL active sequences
		if self._worker.gpu_paged_kv_cache_manager is not None and updated_batch:
			all_global_ids = self._index.local_indices_to_global_seq_ids(updated_batch)
			self._worker.gpu_paged_kv_cache_manager.rebuild_page_table(all_global_ids)
		
		logging.info(
			f"Rank {self.state.rank}: Loaded {len(new_uuids)} new sequences, "
			f"total decode batch now {len(updated_uuids)}"
		)
		
		return updated_uuids, updated_batch

	def load_model(self, max_num_seq: int, comm=None) -> None:
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
		self._worker.deep_free_model_memory()
		self._worker.init_nvshmem()

		# Unified method handles all deployment scenarios
		self.state.model, self._worker.weight_copy_task = self.state.parallel_manager.configure_decoding(
			padding_bsz=max_num_seq, comm=comm
		)
		self._worker.model = self.state.model  # Sync back for _select_tokens etc.
		self._worker.set_phase("decode")
		self.state.core_engine.stop_h2d_worker()
		self.state.core_engine.clear_kv_copy_queue()
		self.state.core_engine.clear_weight_copy_queue()
		self.state.core_engine.reset_decoding_buffer()

		# Only start H2D worker if there are experts to offload
		if self._worker.weight_copy_task.get("routed_expert"):
			self.state.core_engine.set_weight_copy_queue(self._worker.weight_copy_task)
			self.state.core_engine.start_h2d_worker()

		if self.state.rank == 0:
			logging.info(f"[DECODE] Model loaded for decoding phase")


	def config_for_batch(
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
			seq = self.state.global_batch.get_sequence(uuid)
			if seq is None:
				continue
			
			# Compute the correct context length
			# For sequences with decoded tokens: ctx_len = prompt_length + decoded_length
			# For freshly prefilled sequences: ctx_len should equal prompt_length (decoded_length=0)
			expected_ctx = seq.original_prompt_length + seq.decoded_length
			
			# Repair if mismatched
			if seq.current_context_length != expected_ctx:
				old_ctx = seq.current_context_length
				seq.current_context_length = expected_ctx
				ctx_len_repaired_count += 1
				if old_ctx == 0 or abs(old_ctx - expected_ctx) > 100:
					# Only log significant mismatches to avoid log spam
					logging.warning(
						f"Rank {self.state.rank}: Repaired {uuid[:8]} gid={seq.global_idx}: "
						f"ctx_len {old_ctx} → {expected_ctx} (prompt={seq.prompt_length}, decoded={seq.decoded_length})"
					)
		
		if ctx_len_repaired_count > 0:
			logging.info(
				f"Rank {self.state.rank}: Repaired current_context_length for {ctx_len_repaired_count}/{len(decode_uuids)} sequences"
			)
		
		# ============ END CRITICAL FIX ============
		
		# VALIDATION: Verify decode_uuids consistency across all ranks
		local_uuid_count = torch.tensor([len(decode_uuids)], dtype=torch.int64, device=self.state.torch_device)
		all_uuid_counts = [torch.zeros_like(local_uuid_count) for _ in range(self.state.world_size)]
		dist.all_gather(all_uuid_counts, local_uuid_count)
		uuid_counts = [int(t.item()) for t in all_uuid_counts]
		
		if len(set(uuid_counts)) > 1:
			logging.error(
				f"Rank {self.state.rank}: CRITICAL - decode_uuids count mismatch at _config_decoding_for_batch entry! Counts: {uuid_counts}."
			)
		
		# DIAGNOSTIC: Log sequence states at decode config entry
		# This helps identify KV corruption issues during prefill→decode transitions
		resuming_seqs = []
		fresh_seqs = []
		for uuid in decode_uuids:
			seq = self.state.global_batch.get_sequence(uuid)
			
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
				f"Rank {self.state.rank}: _config_decoding_for_batch: "
				f"{len(resuming_seqs)} RESUMING sequences (decoded_length > 0). "
				f"First 5: {resuming_seqs[:5]}"
			)
			# Check for potential issues: sequences with decoded tokens but no GPU reservation flag reset
			problematic = [s for s in resuming_seqs 
						   if s['gpu_pages_allocated'] == 0 and s['had_initial_gpu_reservation']]
			if problematic:
				logging.error(
					f"Rank {self.state.rank}: POTENTIAL BUG: {len(problematic)} sequences have "
					f"decoded_length>0, gpu_pages_allocated=0, but had_initial_gpu_reservation=True! "
					f"First 5: {problematic[:5]}"
				)
		
		if fresh_seqs and self.state.rank == 0 and BATCHGEN_CB_DEBUG:
			logging.debug(
				f"_config_decoding_for_batch: {len(fresh_seqs)} FRESH sequences (decoded_length=0)"
			)

		# ============ SIMPLIFIED: Model and GPU KV manager already initialized ============
		# Model loading and GPU KV manager init now happen in generate() BEFORE batch selection
		# via _load_decode_model() and _init_gpu_kv_with_actual_size()
		assert self.state.model is not None, (
			"Model must be loaded before _config_decoding_for_batch(). "
			"Ensure _load_decode_model() was called first."
		)
		assert self._worker.gpu_paged_kv_cache_manager is not None and self._worker.gpu_paged_kv_cache_manager.is_initialized, (
			"GPU KV manager must be initialized before _config_decoding_for_batch(). "
			"Ensure _init_gpu_kv_with_actual_size() was called first."
		)

		# Allocate GPU KV for sequences
		if local_decode_indices:
			self._worker._allocate_gpu_kv_two_page_buffer(local_decode_indices, load_from_host=True)
			for local_idx in local_decode_indices:
				uuid = self.state.local_to_uuid_map[local_idx]
				seq = self.state.global_batch.get_sequence(uuid)
				seq.gpu_pages_allocated = seq.get_gpu_pages_for_two_page_buffer()
				# Mark initial reservation done
				seq.mark_initial_gpu_reservation_done()
				self.state.sequences_with_gpu_kv.add(uuid)
		
		if self.state.rank == 0:
			logging.info(f"[DECODE] Config completed: {(time.perf_counter() - start_time)*1000:.1f}ms, {len(decode_uuids)} sequences")


	def run_continuous(
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
		if "deepseek" in self.state.model_config.model_type:
			self.state.model.model._use_flash_attention_2 = True
		
		RUNTIME_ATTN_MODE = self._worker.engine_config.Basic_Config.attn_mode
		if RUNTIME_ATTN_MODE != 3:
			self._worker._decoding_legacy_modes(new_tokens, decode_uuids, batch, 1)
			return decode_uuids, batch
		
		# Setup
		gpu_manager = self._worker.gpu_paged_kv_cache_manager
		if gpu_manager is None:
			gpu_manager = getattr(self.state.core_engine, "gpu_paged_kv_manager", None)
		
		worker_view = getattr(self.state.core_engine, "host_paged_kv_worker_view", None)
		
		Attn_Wrapper.gpu_paged_kv_manager = gpu_manager
		Attn_Wrapper.host_paged_kv_worker_view = worker_view
		Attn_Wrapper.scale = scale_dict
		Attn_Wrapper.past_key_states = past_key_states
		Attn_Wrapper.past_value_states = past_value_states
		Attn_Wrapper.cur_batch = self._index.local_indices_to_global_seq_ids(batch) if batch else []

		# Also bind to AttnWrapperBase for models using new wrapper system (e.g., GPT-OSS)
		if isinstance(gpu_manager, DualKVCacheCoordinator):
			AttnWrapperBase.gpu_paged_kv_manager = gpu_manager.primary
			AttnWrapperBase.gpu_paged_kv_manager_aux = gpu_manager.auxiliary
		else:
			AttnWrapperBase.gpu_paged_kv_manager = gpu_manager
			AttnWrapperBase.gpu_paged_kv_manager_aux = None
		AttnWrapperBase.host_paged_kv_worker_view = worker_view
		AttnWrapperBase.cur_batch = Attn_Wrapper.cur_batch

		# CRITICAL FIX: Ensure page table matches cur_batch at entry
		# This fixes order mismatch that can occur during decode→prefill→decode transitions
		if gpu_manager and gpu_manager._gpu_page_table_manager:
			entry_slot_order = list(gpu_manager._gpu_page_table_manager.slot_to_seq_id) if gpu_manager._gpu_page_table_manager.slot_to_seq_id else []
			entry_cur_batch = list(Attn_Wrapper.cur_batch) if Attn_Wrapper.cur_batch else []
			if entry_slot_order != entry_cur_batch:
				logging.error(
					f"Rank {self.state.rank}: ORDER MISMATCH at decoding_continuous entry: "
					f"slot_to_seq_id={entry_slot_order[:5]}{'...' if len(entry_slot_order) > 5 else ''} (len={len(entry_slot_order)}), "
					f"cur_batch={entry_cur_batch[:5]}{'...' if len(entry_cur_batch) > 5 else ''} (len={len(entry_cur_batch)}). Rebuilding page table..."
				)
				# Rebuild page table to match cur_batch order
				if entry_cur_batch:
					gpu_manager.rebuild_page_table(entry_cur_batch)
					logging.info(f"Rank {self.state.rank}: Page table rebuilt to match cur_batch order")
			else:
				if BATCHGEN_CB_DEBUG:
					logging.debug(
						f"Rank {self.state.rank}: decoding_continuous entry OK. "
						f"batch_size={len(batch)}, cur_batch={entry_cur_batch[:5]}{'...' if len(entry_cur_batch) > 5 else ''}"
					)

		# Async state (stored in WorkerState for KVCacheManager access)
		self.state.pending_kv_tasks = []
		self.state.pending_kv_tensors = []
		
		pending_async_task = None
		pending_load_uuids = []
		pending_load_local = []
		pending_load_global = []
		
		# Validation
		for local_idx in batch:
			uuid = self.state.local_to_uuid_map.get(local_idx)
			if uuid and uuid not in self.state.sequences_with_gpu_kv:
				self.state.sequences_with_gpu_kv.add(uuid)
		
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
		global_batch_size = len(self.state.global_batch)

		# ========== INITIAL MOE BUFFER SYNC ==========
		# Sync buffer size BEFORE first forward pass to prevent overflow.
		# The boundary sync (in _page_boundary_fast) only happens after DECISION_INTERVAL
		# iterations, but the first forward pass runs immediately. Without this sync,
		# if one rank has more tokens than the initial estimate (ceil(total/world_size)),
		# we get buffer overflow.
		local_batch_size = torch.tensor([len(batch)], dtype=torch.int64, device=self.state.torch_device)
		dist.all_reduce(local_batch_size, op=dist.ReduceOp.MAX)
		max_batch_size = local_batch_size.item()

		if max_batch_size > 0 and hasattr(self, 'parallel_manager') and self.state.parallel_manager is not None:
			if hasattr(self.state.parallel_manager, 'set_num_tokens_per_rank'):
				self.state.parallel_manager.set_num_tokens_per_rank(max_batch_size)

		# OPTIMIZATION: Track if page table was verified since last batch change
		# Avoids redundant page table checks between boundaries
		_page_table_verified_this_batch = True  # Start True after entry check

		# Main decode loop — enable decode watchdog for monitoring
		self._worker.enable_decode_watchdog()
		while decode_uuids:
			local_iteration += 1
			self._cumulative_decode_iterations += 1

			# Feed watchdogs to prevent timeout during long decoding
			self._worker.feed_watchdog()
			self._worker.feed_decode_watchdog()

			# Page boundary check - use DECISION_INTERVAL (configurable via BATCHGEN_DECISION_FREQUENCY_PAGES)
			if local_iteration - last_boundary >= self._worker.DECISION_INTERVAL:
				last_boundary = local_iteration

				(decode_uuids, batch,
				 pending_async_task, pending_load_uuids,
				 pending_load_local, pending_load_global,
				 timing, watermark_triggered) = self._boundary.page_boundary_fast(
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
					post_boundary_batch_global_ids = self._index.local_indices_to_global_seq_ids(batch)

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
					num_waited = self._kv.wait_pending_tasks()
					if num_waited > 0:
						logging.info(
							f"[WATERMARK-KV-SYNC] Rank {self.state.rank}: Waited for {num_waited} pending KV append tasks "
							f"before putting sequences ON_HOLD"
						)
					
					logging.info(
						f"[WATERMARK] Rank {self.state.rank}: Decode interrupted - putting {len(decode_uuids)} "
						f"sequences ON_HOLD, will trigger prefill"
					)
					# Put all remaining sequences ON_HOLD
					self._worker._rebalancer.put_on_hold(decode_uuids)
					# Exit decode loop - will return to generate() which will trigger prefill
					break
				
				# Detailed logging at every boundary (only rank 0)
				if self.state.rank == 0:
					# Get status counts
					# - in_decode: sequences currently in decode batch (IN_DECODE status)
					# - onhold: sequences paused with host KV (ON_HOLD status)  
					# - prefilled: sequences prefilled but not yet decoding (PREFILLED status)
					# - host_kv_total: total sequences with host KV = prefilled + onhold + in_decode
					num_in_decode = timing.total_active
					num_onhold = len(self.state.global_batch.get_sequences_by_status(SequenceStatus.ON_HOLD))
					num_prefilled = timing.total_prefilled
					num_completed_total = timing.total_completed_cumulative
					num_host_kv_total = num_prefilled + num_onhold + num_in_decode
					
					# Get page stats if available
					page_info = ""
					if self._kv._host_kv_page_stats:
						ps = self._kv._host_kv_page_stats
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
							torch.cuda.synchronize(self.state.torch_device)
						dist.barrier()
						
						decode_uuids, batch = self._worker._finalize_async_load_minimal(
							pending_async_task, pending_load_uuids,
							pending_load_local, pending_load_global,
							decode_uuids, batch, gpu_manager
						)
						self._worker._rebuild_page_table_for_batch(batch, gpu_manager)
						
						if batch:
							new_tokens = self._worker._rebuild_input_tokens(batch)
						
						pending_async_task = None
						pending_load_uuids = []
						pending_load_local = []
						pending_load_global = []
						
						if decode_uuids:
							continue
					break
				
				new_tokens = self._worker._rebuild_input_tokens(batch)
				# DEBUG: Log tokens rebuild after boundary
				if new_tokens.shape[0] != len(batch):
					logging.error(
						f"Rank {self.state.rank}: POST-BOUNDARY new_tokens mismatch! "
						f"batch_size={len(batch)}, new_tokens.shape={new_tokens.shape}"
					)
			
			# Forward pass
			forward_start = time.perf_counter()

			# Pre-compute batch_sequences for use in both forward setup and update loop
			batch_sequences = [self.state.global_batch.get_sequence(self.state.local_to_uuid_map[idx]) for idx in batch] if batch else []

			with torch.inference_mode():
				if batch:
					# Collect context lengths, handling rare edge case of ctx_len == 0
					cache_seqlens = []
					for seq in batch_sequences:
						ctx_len = seq.current_context_length
						if ctx_len == 0:  # Rare edge case - trust prompt_length + decoded_length
							ctx_len = seq.original_prompt_length + seq.decoded_length
							if ctx_len > 0:
								seq.current_context_length = ctx_len
						cache_seqlens.append(ctx_len)

					max_ctx = max(cache_seqlens)
					# Build attention metadata directly on GPU
					seqlens_tensor = torch.tensor(cache_seqlens, dtype=torch.int64, device=self.state.torch_device)

					Attn_Wrapper.attention_mask = None  # Removed: no longer used in decode
					Attn_Wrapper.cache_seqlens = seqlens_tensor.to(torch.int32)
					Attn_Wrapper.position_ids = (Attn_Wrapper.cache_seqlens - 1).unsqueeze(-1).to(torch.int64)
					Attn_Wrapper.max_seqlen = max_ctx

					# CRITICAL: Also bind to AttnWrapperBase for models using new wrapper system (GPT-OSS)
					AttnWrapperBase.attention_mask = None  # Removed: no longer used in decode
					AttnWrapperBase.cache_seqlens = Attn_Wrapper.cache_seqlens
					AttnWrapperBase.position_ids = Attn_Wrapper.position_ids
					AttnWrapperBase.max_seqlen = max_ctx

					if new_tokens.shape[0] != len(batch):
						new_tokens = self._worker._rebuild_input_tokens(batch)
				else:
					Attn_Wrapper.attention_mask = None
					Attn_Wrapper.position_ids = torch.zeros((0, 1), dtype=torch.int64, device=self.state.torch_device)
					Attn_Wrapper.cache_seqlens = torch.zeros((0,), dtype=torch.int32, device=self.state.torch_device)
					Attn_Wrapper.max_seqlen = 0
					Attn_Wrapper.cur_batch = []
					new_tokens = torch.zeros((0, 1), dtype=torch.int64, device=self.state.torch_device)
					# Also bind empty state to AttnWrapperBase for GPT-OSS
					AttnWrapperBase.attention_mask = None
					AttnWrapperBase.position_ids = Attn_Wrapper.position_ids
					AttnWrapperBase.cache_seqlens = Attn_Wrapper.cache_seqlens
					AttnWrapperBase.max_seqlen = 0
					AttnWrapperBase.cur_batch = []
				
				if batch:
					Attn_Wrapper.cur_batch = self._index.local_indices_to_global_seq_ids(batch)
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

				# MoE buffer sync: only needed at decision boundaries (batch size changes).
				# Between boundaries, batch size is constant — skip the all_reduce + .item()
				# CPU-GPU sync that drains the GPU pipeline every step.
				# The sync is done in _page_boundary_fast and at initial setup (line ~7099).
				if getattr(self, '_whole_model_graph', False):
					# Whole-model graph needs globally-synced _max_bs for NCCL bucket matching
					_local_bs = torch.tensor([len(batch)], dtype=torch.int64, device=self.state.torch_device)
					dist.all_reduce(_local_bs, op=dist.ReduceOp.MAX)
					_max_bs = max(_local_bs.item(), 1)
					if hasattr(self, 'parallel_manager') and self.state.parallel_manager is not None:
						if hasattr(self.state.parallel_manager, 'set_num_tokens_per_rank'):
							self.state.parallel_manager.set_num_tokens_per_rank(_max_bs)
				else:
					# Per-layer graph or eager: no NCCL in graph, use local batch size
					_max_bs = max(len(batch), 1)

				# KV append callback — deferred: accumulate during forward, single sync after
				current_batch = list(batch)
				_kv_worker_view = getattr(self.state.core_engine, "host_paged_kv_worker_view", None)

				if _kv_worker_view is not None:
					_kv_seq_ids = []
					_kv_seq_lengths = []
					for local_idx in current_batch:
						uuid = self.state.local_to_uuid_map[local_idx]
						seq = self.state.global_batch.get_sequence(uuid)
						_kv_seq_ids.append(seq.global_idx)
						_kv_seq_lengths.append(seq.current_context_length - 1)
					self.state.deferred_kv_batch = (_kv_seq_ids, _kv_seq_lengths)
					self.state.deferred_kv_entries = []
					self.state.deferred_kv_worker_view = _kv_worker_view

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
						torch.cuda.synchronize()
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
						self.state.deferred_kv_entries.append((layer_idx, k_tensor, v_tensor))

				Attn_Wrapper.kv_append_callback = kv_append_callback
				# Also bind to AttnWrapperBase for models using new wrapper system (e.g., GPT-OSS)
				AttnWrapperBase.kv_append_callback = kv_append_callback

				# Forward
				_use_graph = (
					getattr(self, '_whole_model_graph', False)
					and self._cuda_graph_manager is not None
					and _max_bs <= self._whole_model_bucketing._max_bucket
				)
				if _use_graph:
					# Whole-model CUDA graph replay.
					# CRITICAL: Use _max_bs (globally-synced max batch size) for bucket
					# computation, NOT local len(batch). The graph has NCCL all_reduce
					# baked inside — all ranks MUST replay the same bucket's graph,
					# otherwise mismatched NCCL ops cause deadlock.
					batch_size = len(batch)
					bucket = self._whole_model_bucketing.get_padded_size(_max_bs)
					page_table_tensor = gpu_manager._gpu_page_table_manager.gpu_table
					slot_indices_tensor = gpu_manager._gpu_page_table_manager._slot_index_tensor
					if slot_indices_tensor is None:
						# Rebuild may have cleared it; reconstruct as simple arange
						slot_indices_tensor = torch.arange(
							page_table_tensor.shape[0], dtype=torch.int32,
							device=self.state.torch_device,
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
					new_tokens_out = self._worker._select_tokens(logits)

					# Fire KV host offload callbacks for all layers.
					# KV buffers are static-address tensors written inside the graph;
					# clone before passing to async D2H to protect tensor lifespan.
					kv_cb = getattr(AttnWrapperBase, 'kv_append_callback', None)
					wm_seg = getattr(self, '_whole_model_segment', None)
					if kv_cb is not None and wm_seg is not None and wm_seg._kv_buffers is not None:
						for layer_idx in range(wm_seg.num_layers):
							kv_buf = wm_seg._kv_buffers[layer_idx]
							# K2.5 MLA has no separate V cache — pass None for v_tensor
							v_buf = kv_buf.get("value")
							v_clone = v_buf[:batch_size].clone() if v_buf is not None and v_buf.numel() > 0 and not getattr(wm_seg, '_no_v_cache', False) else None
							kv_cb(
								layer_idx,
								kv_buf["key"][:batch_size].clone(),
								v_clone,
							)
				else:
					# Per-layer graph or eager forward
					# CRITICAL: Pass position_ids to model to ensure correct RoPE positioning during decode.
					# Without this, the model generates position_ids = [[0]] for all decode steps,
					# causing RoPE to be applied at position 0 instead of the actual token position.
					outputs = self.state.model(
						new_tokens,
						attention_mask=Attn_Wrapper.attention_mask,
						position_ids=Attn_Wrapper.position_ids,
						use_cache=False
					)
					new_tokens_out = self._worker._select_tokens(outputs.logits[:, -1, :])

			new_tokens = new_tokens_out

			# Flush deferred KV entries — single sync for all layers
			self._kv.flush_deferred_kv()

			# Optimization: Single GPU→CPU transfer for all tokens (vs N transfers in loop)
			# This avoids N GPU synchronizations which cause heavy CPU overhead
			new_tokens_cpu = new_tokens.cpu()

			# Update sequences (reuse batch_sequences from forward pass setup)
			for i, (local_idx, seq) in enumerate(zip(batch, batch_sequences)):
				if self._worker._is_sequence_completed(seq):
					continue

				decode_pos = seq.decoded_length
				if BATCHGEN_CB_DEBUG:
					qb_ptr = self._worker.query_book[local_idx].decoded_tokens.data_ptr()
					seq_ptr = seq.decoded_tokens.data_ptr()
					if qb_ptr != seq_ptr:
						logging.error(
							f"Rank {self.state.rank}: query_book/seq decoded_tokens MISMATCH for "
							f"local_idx={local_idx}, uuid={seq.uuid[:8]}, "
							f"qb_ptr={qb_ptr:#x}, seq_ptr={seq_ptr:#x}"
						)
				self._worker.query_book[local_idx].decoded_tokens[:, decode_pos] = new_tokens_cpu[i]

				seq.decoded_length += 1
				seq.current_context_length += 1

				# Use CPU tensor to avoid GPU sync
				if self._worker._should_stop_at_eos(new_tokens_cpu[i].item()):
					seq.eos_reached = True

				if seq.decoded_length >= seq.max_decode_length:
					seq.eos_reached = True

			self._cumulative_forward_ms += (time.perf_counter() - forward_start) * 1000

		# Cleanup
		self._kv.wait_pending_tasks()
		if pending_async_task is not None:
			pending_async_task.wait()
			torch.cuda.synchronize(self.state.torch_device)

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
		AttnWrapperBase.kv_append_callback = None
		
		# Summary (uses cumulative counters for accurate cross-round totals)
		# Only show when BATCHGEN_CB_LOG=DEBUG
		if self.state.rank == 0 and self._cumulative_decode_boundaries > 0 and BATCHGEN_CB_DEBUG:
			avg_forward = self._cumulative_forward_ms / self._cumulative_decode_iterations if self._cumulative_decode_iterations > 0 else 0
			avg_round = self._cumulative_boundary_ms / self._cumulative_decode_boundaries
			logging.debug(
				f"\n{'='*50}\n"
				f"DECODE SUMMARY (Rank 0)\n"
				f"{'='*50}\n"
				f"Total Iterations: {self._cumulative_decode_iterations}, Total Rounds: {self._cumulative_decode_boundaries}\n"
				f"Avg forward: {avg_forward:.2f}ms\n"
				f"Avg round overhead: {avg_round:.2f}ms\n"
				f"Round overhead/token: {avg_round / self._worker.DECISION_INTERVAL:.3f}ms\n"
				f"{'='*50}"
			)

		self._worker.disable_decode_watchdog()
		return decode_uuids, batch
