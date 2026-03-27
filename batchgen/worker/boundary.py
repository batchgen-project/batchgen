"""
BoundaryHandler: Page boundary decisions and execution for decode phase.

Extracted from batchgen_worker.py (Step 8 of scheduler split).
Handles: consolidated collective boundary operations, rank-0 decision making,
GPU page extension, completion handling, async sequence loading.
"""
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from batchgen.worker.state import WorkerState
from batchgen.kv_cache.gpu_paged_kv_manager import GPUPagedKVCacheManager
from batchgen.continuous_batching import (
	BoundaryDecisions,
	FastBoundaryTimingStats,
	select_sequences_for_eviction as select_host_kv_eviction,
	EvictionStrategy,
)
from batchgen.sequence import SequenceStatus

BATCHGEN_CB_DEBUG = os.environ.get("BATCHGEN_CB_LOG", "").upper() == "DEBUG"


class BoundaryHandler:
	"""Page boundary decisions and execution for the decode phase.

	Uses a worker reference for shared methods (finalize_async_load,
	extend_gpu_kv, rebuild_page_table, etc.) that are tightly coupled
	to the worker's GPU management.
	"""

	def __init__(self, state: WorkerState, index_manager, kv_manager, worker):
		self.state = state
		self._index = index_manager
		self._kv = kv_manager
		self._worker = worker
		self._boundary_count = 0

	def compute_decisions(
		self,
		decode_uuids: List[str],
		global_seq_state: Dict[str, Dict],
		global_candidate_info: Dict[str, Dict],
		per_rank_free: List[int],
		chunk_size: int,
		worker_view: Optional[object],
	) -> 'BoundaryDecisions':
		"""Compute ALL batching decisions on rank 0 only.

		This method is called ONLY by rank 0. The returned BoundaryDecisions
		struct is broadcast to all ranks, which then execute their local portion.

		This centralizes all decision-making to prevent desync between ranks.
		"""
		# Identify completed sequences
		completed_uuids = []
		active_uuids = []
		for uuid in decode_uuids:
			state = global_seq_state.get(uuid)
			if state and state['completed']:
				completed_uuids.append(uuid)
			else:
				active_uuids.append(uuid)

		# Host KV growth decisions
		host_growth_uuids = []
		host_growth_pages_list = []
		total_growth_needed = 0
		for uuid in active_uuids:
			state = global_seq_state.get(uuid)
			if state and state.get('needs_host_growth'):
				growth_pages = state.get('host_growth_pages', 0)
				if growth_pages > 0:
					host_growth_uuids.append(uuid)
					host_growth_pages_list.append(growth_pages)
					total_growth_needed += growth_pages

		growth_feasible = False
		if total_growth_needed > 0 and worker_view is not None:
			host_stats = worker_view.get_stats()
			safety_margin = int(host_stats.num_total_pages * 0.05)
			growth_feasible = total_growth_needed <= (host_stats.num_free_pages - safety_margin)
			if not growth_feasible:
				logging.warning(
					f"[HOST_KV_GROWTH] Skipped: need {total_growth_needed} pages "
					f"but only {host_stats.num_free_pages} free "
					f"({safety_margin} reserved). Will rely on eviction or sequence completions to free pages."
				)

		# Host KV eviction decisions
		host_evicted_uuids = []
		decode_after_eviction = list(active_uuids)
		if self.state.enable_host_kv_eviction and active_uuids and worker_view is not None:
			host_stats = worker_view.get_stats()
			total_pages = host_stats.num_total_pages
			free_pages = host_stats.num_free_pages
			free_pct = (free_pages / total_pages * 100) if total_pages > 0 else 100

			if free_pct < self.state.host_kv_eviction_watermark:
				eviction_candidates = []
				completed_set = set(completed_uuids)
				for uuid in active_uuids:
					state = global_seq_state.get(uuid)
					if state and uuid not in completed_set:
						eviction_candidates.append((uuid, {
							'decoded_length': state['decoded_length'],
							'host_pages_allocated': state.get('host_pages_allocated', 0),
							'global_idx': self.state.global_batch.get_sequence(uuid).global_idx,
						}))

				target_free = int(total_pages * self.state.host_kv_eviction_watermark / 100)
				pages_to_free = max(0, target_free - free_pages)

				if pages_to_free > 0 and eviction_candidates:
					host_evicted_uuids, _ = select_host_kv_eviction(
						eviction_candidates, pages_to_free,
						strategy=EvictionStrategy.SHORTEST_FIRST,
						page_key='host_pages_allocated',
					)
					if host_evicted_uuids:
						evicted_set = set(host_evicted_uuids)
						decode_after_eviction = [u for u in active_uuids if u not in evicted_set]

		# GPU page extension / on-hold decisions
		seqs_needing_extension = []
		total_additional_by_rank = [0] * self.state.world_size

		for uuid in decode_after_eviction:
			state = global_seq_state.get(uuid)
			if state and state['additional_pages_needed'] > 0:
				assigned_rank = state['assigned_rank']
				total_additional_by_rank[assigned_rank] += state['additional_pages_needed']
				seqs_needing_extension.append(uuid)

		all_can_extend = all(
			total_additional_by_rank[r] <= per_rank_free[r]
			for r in range(self.state.world_size)
		)

		onhold_uuids = []
		actual_extension_by_rank = [0] * self.state.world_size

		if all_can_extend:
			actual_extension_by_rank = list(total_additional_by_rank)
		elif not all_can_extend:
			for r in range(self.state.world_size):
				if total_additional_by_rank[r] > per_rank_free[r]:
					rank_seqs = [
						(uuid, global_seq_state[uuid])
						for uuid in decode_after_eviction
						if uuid in global_seq_state and global_seq_state[uuid]['assigned_rank'] == r
					]
					rank_seqs.sort(
						key=lambda x: (x[1]['decoded_length'],
									self.state.global_batch.get_sequence(x[0]).global_idx)
					)
					pages_to_free = total_additional_by_rank[r] - per_rank_free[r]
					freed = 0
					for uuid, state in rank_seqs:
						if freed >= pages_to_free:
							break
						onhold_uuids.append(uuid)
						freed += state['gpu_pages_allocated']

			# Compute actual extension for remaining sequences
			onhold_set = set(onhold_uuids)
			for uuid in seqs_needing_extension:
				if uuid not in onhold_set:
					state = global_seq_state.get(uuid, {})
					r = state.get('assigned_rank')
					if r is not None:
						actual_extension_by_rank[r] += state.get('additional_pages_needed', 0)

		# Load candidate selection
		onhold_set = set(onhold_uuids)
		completed_set = set(completed_uuids)
		evicted_set = set(host_evicted_uuids)
		decode_uuids_final = [u for u in decode_after_eviction if u not in onhold_set]

		new_load_uuids = []
		if global_candidate_info and decode_uuids_final:
			load_candidates_synced = sorted(
				[u for u in global_candidate_info.keys()
				 if u not in completed_set and u not in onhold_set and u not in evicted_set],
				key=lambda u: (
					-global_candidate_info[u].get('decoded_length', 0),
					self.state.global_batch.get_sequence(u).global_idx if self.state.global_batch.get_sequence(u) else float('inf')
				)
			)

			# Compute adjusted free pages after extensions (arithmetic, no collective needed)
			adjusted_per_rank_free = [
				per_rank_free[r] - actual_extension_by_rank[r]
				for r in range(self.state.world_size)
			]

			rank_pages_used = [0] * self.state.world_size
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

		return BoundaryDecisions(
			completed_uuids=completed_uuids,
			active_uuids=active_uuids,
			host_growth_uuids=host_growth_uuids,
			host_growth_pages=host_growth_pages_list,
			growth_feasible=growth_feasible,
			host_evicted_uuids=host_evicted_uuids,
			onhold_uuids=onhold_uuids,
			seqs_needing_extension=seqs_needing_extension,
			new_load_uuids=new_load_uuids,
			decode_uuids_final=decode_uuids_final,
		)

	def page_boundary_fast(
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
		timing.num_kv_append_tasks = self._kv.wait_pending_tasks()
		timing.wait_kv_append_ms = (time.perf_counter() - t0) * 1000
		
		# decode_uuids sync: only run in debug mode for desync detection.
		# In production, rank 0 makes all decisions so sync is unnecessary.
		t_sync = time.perf_counter()
		if BATCHGEN_CB_DEBUG:
			local_decode_set = set(decode_uuids)
			all_decode_sets = [None] * self.state.world_size
			dist.all_gather_object(all_decode_sets, local_decode_set)
			all_sets_equal = all(s == local_decode_set for s in all_decode_sets if s is not None)
			if not all_sets_equal:
				for r, s in enumerate(all_decode_sets):
					if s != local_decode_set:
						diff_in_r = s - local_decode_set if s else set()
						diff_in_local = local_decode_set - s if s else local_decode_set
						logging.error(
							f"Rank {self.state.rank}: decode_uuids DESYNC detected at boundary start! "
							f"Rank {r} has {len(diff_in_r)} extra: {list(diff_in_r)[:5]}, "
							f"Rank {self.state.rank} has {len(diff_in_local)} extra: {list(diff_in_local)[:5]}"
						)
				# Use RANK 0 as authoritative source
				rank0_set = all_decode_sets[0] if all_decode_sets[0] is not None else set()
				decode_uuids = sorted(
					rank0_set,
					key=lambda u: self.state.global_batch.get_sequence(u).global_idx if self.state.global_batch.get_sequence(u) else float('inf')
				)
				batch = self._index.get_local_indices_for_uuids(decode_uuids)
				logging.warning(f"Rank {self.state.rank}: Using rank-0 authoritative set at boundary start, decode_uuids now {len(decode_uuids)}")
		timing.sync_decode_uuids_ms = (time.perf_counter() - t_sync) * 1000
		
		# Integrate previous async load if any
		if pending_load_uuids:  # ALL ranks have identical pending_load_uuids
			t0 = time.perf_counter()

			if BATCHGEN_CB_DEBUG:
				logging.debug(
					f"Rank {self.state.rank}: Integrating {len(pending_load_uuids)} async-loaded sequences"
				)

			if pending_async_load_task is not None:
				pending_async_load_task.wait()
				torch.cuda.synchronize(self.state.torch_device)

			timing.wait_async_load_ms = (time.perf_counter() - t0) * 1000

			# barrier ensures all ranks finish async load before continuing
			dist.barrier()
			
			t0 = time.perf_counter()
			decode_uuids, batch = self._worker._finalize_async_load_minimal(
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
				self._worker._rebuild_page_table_for_batch(batch, gpu_manager)
				# Verify page table matches batch, fix if needed
				if gpu_manager._gpu_page_table_manager:
					post_finalize_slot_order = list(gpu_manager._gpu_page_table_manager.slot_to_seq_id) if gpu_manager._gpu_page_table_manager.slot_to_seq_id else []
					post_finalize_batch_global_ids = self._index.local_indices_to_global_seq_ids(batch)
					if post_finalize_slot_order != post_finalize_batch_global_ids:
						gpu_manager.rebuild_page_table(post_finalize_batch_global_ids)

		if not decode_uuids:
			timing.total_ms = (time.perf_counter() - boundary_start) * 1000
			return decode_uuids, batch, None, [], [], [], timing, False
		
		# ========== PHASE 1: SINGLE BATCHED ALL_GATHER ==========
		t0 = time.perf_counter()
		
		local_free_pages = gpu_manager.get_stats().num_free_pages if gpu_manager and gpu_manager.is_initialized else 0
		
		# DEBUG: Log decode_uuids and which ones this rank owns
		my_owned = [u for u in decode_uuids if u in self.state.uuid_to_local_map]
		if self.state.rank == 0:
			logging.debug(
				f"Rank {self.state.rank}: State gathering - decode_uuids_len={len(decode_uuids)}, "
				f"my_owned_count={len(my_owned)}"
			)
		
		# Build local state for sequences owned by this rank
		chunk_size = self._worker._get_effective_chunk_size()
		local_seq_state = {}
		for uuid in decode_uuids:
			if uuid in self.state.uuid_to_local_map:
				seq = self.state.global_batch.get_sequence(uuid)
				is_completed = self._worker._is_sequence_completed(seq)
				local_seq_state[uuid] = {
					'decoded_length': seq.decoded_length,
					'current_context_length': seq.current_context_length,
					'gpu_pages_allocated': seq.gpu_pages_allocated,
					'eos_reached': seq.eos_reached,
					'completed': is_completed,
					'additional_pages_needed': seq.get_additional_gpu_pages_needed(),
					'assigned_rank': seq.assigned_rank,  # Include for consistency
					# Host KV growth fields
					'needs_host_growth': seq.needs_host_kv_growth(chunk_size),
					'host_growth_pages': seq.get_host_growth_pages(chunk_size),
					'host_pages_allocated': seq.host_pages_allocated,
					'host_token_capacity': seq.host_token_capacity,
				}
		
		# Get candidates for loading - report PREFILLED/ON_HOLD sequences that could be loaded
		# CRITICAL FIX: Only report PREFILLED or ON_HOLD sequences as load candidates.
		# QUEUEING sequences have NOT been registered with host KV yet (registration
		# happens during _config_prefill_for_batch), so trying to load them would fail
		# with "Sequence X is not registered" error from the host KV backend.
		decode_uuids_set = set(decode_uuids)
		local_candidate_state = {}
		valid_load_statuses = {SequenceStatus.PREFILLED, SequenceStatus.ON_HOLD}
		for uuid in self.state.uuid_to_local_map.keys():
			if uuid in decode_uuids_set:
				continue  # Already in decode batch
			seq = self.state.global_batch.get_sequence(uuid)
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
		
		all_payloads = [None] * self.state.world_size
		dist.all_gather_object(all_payloads, local_payload)
		
		timing.gather_ms = (time.perf_counter() - t0) * 1000
		
		# ========== PHASE 2: MERGE GATHERED DATA + RANK-0 DECISIONS ==========
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
			for missing_uuid in missing_uuids[:10]:
				seq = self.state.global_batch.get_sequence(missing_uuid)
				expected_rank = seq.assigned_rank if seq else "N/A"
				in_local_map = missing_uuid in self.state.uuid_to_local_map
				seq_status = seq.status.name if seq else "NOT_FOUND"
				rank_reported = [r for r, p in enumerate(all_payloads)
								if p and p.get('seq_state', {}).get(missing_uuid)]
				logging.error(
					f"Rank {self.state.rank}: Missing UUID={missing_uuid}, assigned_rank={expected_rank}, "
					f"in_local_map={in_local_map}, status={seq_status}, reported_by_ranks={rank_reported}"
				)
			logging.error(
				f"Rank {self.state.rank}: CRITICAL - {len(missing_uuids)} sequences missing from gathered state! "
				f"decode_uuids_len={len(decode_uuids)}, global_seq_state_len={len(global_seq_state)}, "
				f"Missing first 5: {missing_uuids[:5]}"
			)
			decode_uuids = [u for u in decode_uuids if u in global_seq_state]

		# Update local SequenceEntry with gathered info (for sequences on other ranks)
		for uuid, state in global_seq_state.items():
			if uuid not in self.state.uuid_to_local_map:
				seq = self.state.global_batch.get_sequence(uuid)
				if seq is not None:
					seq.decoded_length = state['decoded_length']
					seq.current_context_length = state['current_context_length']
					seq.gpu_pages_allocated = state['gpu_pages_allocated']
					seq.eos_reached = state['eos_reached']
					# Sync host KV fields to keep all ranks consistent for migration planning
					seq.host_pages_allocated = state['host_pages_allocated']
					seq.host_token_capacity = state['host_token_capacity']

		# ========== RANK 0 COMPUTES ALL DECISIONS ==========
		# Only rank 0 makes batching decisions. All other ranks receive via broadcast.
		# This eliminates desync from independent decision-making.
		worker_view = getattr(self.state.core_engine, "host_paged_kv_worker_view", None)

		if self.state.rank == 0:
			decisions = self.compute_decisions(
				decode_uuids, global_seq_state, global_candidate_info,
				per_rank_free, chunk_size, worker_view,
			)
		else:
			decisions = None

		# ========== PHASE 3: BROADCAST DECISIONS ==========
		decisions_list = [decisions]
		dist.broadcast_object_list(decisions_list, src=0)
		decisions = decisions_list[0]

		timing.num_completed = len(decisions.completed_uuids)
		timing.num_onhold = len(decisions.onhold_uuids)

		# ========== PHASE 4: EXECUTE DECISIONS LOCALLY ==========
		# All ranks execute the same decisions, but only operate on locally-owned sequences

		# A. Release completed sequences
		completed_uuids = decisions.completed_uuids
		if completed_uuids:
			self._worker._update_batch_status(completed_uuids, SequenceStatus.COMPLETED)
			# Incremental write: gather completed tokens to rank 0
			self._worker._submit_completed_to_incremental_writer(completed_uuids)
			my_completed = [u for u in completed_uuids if u in self.state.uuid_to_local_map]
			if my_completed:
				my_completed_local = self._index.get_local_indices_for_uuids(my_completed)
				self._kv.release_gpu_pages(my_completed_local)
				self._worker._release_host_kv_pages_for_batch(my_completed)
			# Report completions to adaptive chunk sizer
			if self._worker.adaptive_chunk_sizer is not None:
				for uuid in completed_uuids:
					state = global_seq_state.get(uuid)
					if state:
						self._worker.adaptive_chunk_sizer.report_completion(state['decoded_length'])
			# Log completion details for diagnostics
			if self.state.rank == 0 and BATCHGEN_CB_DEBUG:
				for uuid in completed_uuids:
					seq = self.state.global_batch.get_sequence(uuid)
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
		batch = self._index.get_local_indices_for_uuids(decode_uuids)

		# B. Host KV growth
		if decisions.growth_feasible and decisions.host_growth_uuids:
			host_grow_requests = []
			for uuid, growth_pages in zip(decisions.host_growth_uuids, decisions.host_growth_pages):
				# Update metadata on ALL ranks (decisions are broadcast from rank 0).
				# This keeps host_pages_allocated consistent across ranks, which is
				# critical for deterministic migration planning in _plan_kv_migration().
				seq = self.state.global_batch.get_sequence(uuid)
				seq.host_token_capacity += growth_pages * seq.PAGE_SIZE
				seq.host_pages_allocated += growth_pages
				# Only do actual host page allocation on owner rank
				if uuid in self.state.uuid_to_local_map:
					host_grow_requests.append((seq.global_idx, growth_pages))

			if host_grow_requests and worker_view is not None:
				worker_view.grow_pages_for_sequences(host_grow_requests)
				if self.state.rank == 0:
					logging.debug(
						f"[HOST_KV_GROWTH] Grew {len(host_grow_requests)} sequences, "
						f"chunk_size={chunk_size}"
					)
				if self.state.rank == 0 and BATCHGEN_CB_DEBUG:
					for uuid, growth_pages in zip(decisions.host_growth_uuids, decisions.host_growth_pages):
						if uuid in self.state.uuid_to_local_map:
							seq = self.state.global_batch.get_sequence(uuid)
							old_cap = seq.host_token_capacity - growth_pages * seq.PAGE_SIZE
							runway = seq.host_token_capacity - seq.current_context_length
							logging.debug(
								f"[HOST_KV_GROWTH_DETAIL] seq={uuid[:8]} "
								f"old_cap={old_cap} new_cap={seq.host_token_capacity} "
								f"runway={runway} pages={growth_pages}"
							)

		# C. Host KV eviction
		host_evicted_uuids = decisions.host_evicted_uuids
		if host_evicted_uuids:
			my_evicted = [u for u in host_evicted_uuids if u in self.state.uuid_to_local_map]
			if my_evicted:
				my_evicted_local = self._index.get_local_indices_for_uuids(my_evicted)
				self._kv.release_gpu_pages(my_evicted_local)
				for uuid in my_evicted:
					seq = self.state.global_batch.get_sequence(uuid)
					if seq.original_prompt_length == seq.prompt_length:
						pass  # Already correct from init
					prompt_tokens = seq.input_ids[0, :seq.prompt_length]
					if seq.decoded_tokens is not None and seq.decoded_length > 0:
						decoded = seq.decoded_tokens[0, :seq.decoded_length]
						seq.evicted_token_ids = torch.cat([prompt_tokens, decoded])
					else:
						seq.evicted_token_ids = prompt_tokens.clone()
					seq.total_decoded_before_eviction = seq.decoded_length
					if BATCHGEN_CB_DEBUG:
						logging.debug(
							f"[HOST_KV_EVICT_DETAIL] seq={uuid[:8]} "
							f"decoded={seq.decoded_length} "
							f"host_pages={seq.host_pages_allocated} "
							f"tokens_saved={len(seq.evicted_token_ids)}"
						)
				evicted_global_ids = [
					self.state.global_batch.get_sequence(u).global_idx for u in my_evicted
				]
				if worker_view is not None:
					worker_view.release_sequence_pages(evicted_global_ids)
					worker_view.unregister_sequences(evicted_global_ids)

			# Update status (all ranks)
			for uuid in host_evicted_uuids:
				seq = self.state.global_batch.get_sequence(uuid)
				seq.gpu_pages_allocated = 0
				seq.host_pages_allocated = 0
				seq.host_token_capacity = 0
				self.state.sequences_with_gpu_kv.discard(uuid)
				self.state.global_batch.update_status(uuid, SequenceStatus.EVICTED)

			evicted_set = set(host_evicted_uuids)
			decode_uuids = [u for u in decode_uuids if u not in evicted_set]
			batch = self._index.get_local_indices_for_uuids(decode_uuids)

			if self.state.rank == 0:
				logging.info(
					f"[HOST_KV_EVICT] Evicted {len(host_evicted_uuids)} sequences"
				)

		timing.process_ms = (time.perf_counter() - t0) * 1000

		# Calculate completed count BEFORE early return to ensure final iteration reports correctly
		timing.total_completed_cumulative = len(self.state.global_batch.get_sequences_by_status(SequenceStatus.COMPLETED))

		if not decode_uuids:
			timing.total_ms = (time.perf_counter() - boundary_start) * 1000
			return decode_uuids, batch, None, [], [], [], timing, False

		# D. GPU page extension / on-hold (using rank-0 decisions)
		t0 = time.perf_counter()
		onhold_uuids = decisions.onhold_uuids
		onhold_set = set(onhold_uuids)

		if onhold_uuids:
			my_onhold = [u for u in onhold_uuids if u in self.state.uuid_to_local_map]
			if my_onhold:
				local_indices = self._index.get_local_indices_for_uuids(my_onhold)
				global_ids = self._index.local_indices_to_global_seq_ids(local_indices)
				if global_ids and gpu_manager:
					gpu_manager.free_pages_for_sequences(global_ids)
				for uuid in my_onhold:
					seq = self.state.global_batch.get_sequence(uuid)
					seq.gpu_pages_allocated = 0
					self.state.sequences_with_gpu_kv.discard(uuid)

			for uuid in onhold_uuids:
				self.state.global_batch.update_status(uuid, SequenceStatus.ON_HOLD)

			decode_uuids = [u for u in decode_uuids if u not in onhold_set]
			batch = self._index.get_local_indices_for_uuids(decode_uuids)

			if BATCHGEN_CB_DEBUG:
				logging.info(
					f"Rank {self.state.rank}: After on-hold: batch_size={len(batch)}, "
					f"num_onhold={len(onhold_uuids)}, my_onhold={len(my_onhold)}"
				)

		# Extend GPU pages for sequences that need it (not on-hold)
		seqs_needing_extension = decisions.seqs_needing_extension
		remaining_needing_ext = [u for u in seqs_needing_extension if u not in onhold_set]
		my_remaining_ext = [u for u in remaining_needing_ext if u in self.state.uuid_to_local_map]
		if my_remaining_ext:
			self._worker._extend_gpu_kv_allocation(my_remaining_ext)

		timing.extension_ms = (time.perf_counter() - t0) * 1000

		# E. Async load (using rank-0 decisions)
		t0 = time.perf_counter()
		new_async_task = None
		new_load_uuids = decisions.new_load_uuids
		new_load_local = []
		new_load_global = []

		if new_load_uuids and decode_uuids:
			my_new_uuids = [u for u in new_load_uuids
						if global_candidate_info.get(u, {}).get('assigned_rank') == self.state.rank]
			new_load_local = self._index.get_local_indices_for_uuids(my_new_uuids)

			if new_load_local:
				actual_free = gpu_manager.get_stats().num_free_pages if gpu_manager and gpu_manager.is_initialized else 0

				filtered_local = []
				filtered_global = []
				filtered_tokens = []
				pages_used = 0

				for local_idx in new_load_local:
					uuid = self.state.local_to_uuid_map[local_idx]
					seq = self.state.global_batch.get_sequence(uuid)
					pages_needed = seq.get_gpu_pages_for_two_page_buffer()

					if pages_used + pages_needed <= actual_free:
						filtered_local.append(local_idx)
						filtered_global.append(seq.global_idx)
						filtered_tokens.append(pages_needed * self._worker.PAGE_SIZE)
						pages_used += pages_needed
					else:
						logging.warning(
							f"Rank {self.state.rank}: Dropping {uuid[:8]} from load - "
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
						existing_global_ids = self._index.local_indices_to_global_seq_ids(batch)
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

						self._worker._async_load_tensors = {
							'k_ptrs': k_ptrs, 'v_ptrs': v_ptrs,
							'sequence_tensor': sequence_tensor,
							'active_page_counts': active_page_counts,
						}
					timing.load_launch_ms = (time.perf_counter() - t_launch) * 1000
				else:
					new_load_local = []
					new_load_global = []
					logging.warning(
						f"Rank {self.state.rank}: All load candidates dropped due to insufficient pages, "
						f"actual_free={actual_free}"
					)
		
		timing.num_loaded = len(new_load_uuids)
		
		# ========== FINAL PAGE TABLE REBUILD ==========
		t0 = time.perf_counter()
		if BATCHGEN_CB_DEBUG:
			global_ids_for_rebuild = self._index.local_indices_to_global_seq_ids(batch) if batch else []
			logging.debug(
				f"Rank {self.state.rank}: FINAL REBUILD: batch_size={len(batch)}, "
				f"global_ids_count={len(global_ids_for_rebuild)}"
			)
		self._worker._rebuild_page_table_for_batch(batch, gpu_manager)
		if BATCHGEN_CB_DEBUG and gpu_manager and gpu_manager.is_initialized:
			mgr = gpu_manager._gpu_page_table_manager
			if mgr and mgr.gpu_table is not None:
				logging.debug(
					f"Rank {self.state.rank}: After rebuild: gpu_table.shape={mgr.gpu_table.shape}, "
					f"slot_to_seq_id_len={len(mgr.slot_to_seq_id)}"
				)
		timing.rebuild_ms = (time.perf_counter() - t0) * 1000
		
		# ========== UPDATE MOE BUFFER SIZE ==========
		# Find max batch size across all ranks to minimize all-gather/all-reduce communication
		t0 = time.perf_counter()
		local_batch_size = torch.tensor([len(batch)], dtype=torch.int64, device=self.state.torch_device)
		dist.all_reduce(local_batch_size, op=dist.ReduceOp.MAX)
		max_batch_size = local_batch_size.item()
		
		# Update MoE layers with the actual max batch size for this page
		if max_batch_size > 0 and self.state.parallel_manager is not None:
			if hasattr(self.state.parallel_manager, 'set_num_tokens_per_rank'):
				self.state.parallel_manager.set_num_tokens_per_rank(max_batch_size)
		timing.moe_buffer_update_ms = (time.perf_counter() - t0) * 1000
		
		# ========== SINGLE FINAL BARRIER ==========
		t0 = time.perf_counter()
		dist.barrier()
		timing.barrier_ms = (time.perf_counter() - t0) * 1000
		
		# ========== COLLECT STATUS COUNTS ==========
		timing.total_active = len(decode_uuids)
		timing.total_prefilled = len(self.state.global_batch.get_sequences_by_status(SequenceStatus.PREFILLED))
		timing.total_completed_cumulative = len(self.state.global_batch.get_sequences_by_status(SequenceStatus.COMPLETED))
		
		# ========== VERIFY BATCH CONSISTENCY ==========
		# CRITICAL: Ensure batch matches decode_uuids for THIS rank
		expected_local = self._index.get_local_indices_for_uuids(decode_uuids)
		if set(batch) != set(expected_local):
			logging.error(
				f"Rank {self.state.rank}: BATCH MISMATCH after boundary! "
				f"batch={sorted(batch)}, expected={sorted(expected_local)}"
			)
			batch = expected_local  # Fix the batch
			# CRITICAL: Rebuild page table to match the corrected batch
			self._worker._rebuild_page_table_for_batch(batch, gpu_manager)
			logging.info(f"Rank {self.state.rank}: Page table rebuilt after batch correction")
		
		# FINAL VERIFICATION: Ensure page table matches batch before returning
		if batch and gpu_manager and gpu_manager.is_initialized:
			mgr = gpu_manager._gpu_page_table_manager
			if mgr and mgr.gpu_table is not None:
				if mgr.gpu_table.shape[0] != len(batch):
					logging.error(
						f"Rank {self.state.rank}: CRITICAL - Page table STILL mismatched at function return! "
						f"gpu_table.shape[0]={mgr.gpu_table.shape[0]}, batch_size={len(batch)}"
					)
		
		timing.total_ms = (time.perf_counter() - boundary_start) * 1000

		# Periodic host KV diagnostic summary
		self._boundary_count += 1
		if self.state.rank == 0 and BATCHGEN_CB_DEBUG and self._boundary_count % 10 == 0:
			worker_view = getattr(self.state.core_engine, "host_paged_kv_worker_view", None)
			if worker_view is not None:
				hs = worker_view.get_stats()
				used = hs.num_total_pages - hs.num_free_pages
				pct = (used / hs.num_total_pages * 100) if hs.num_total_pages > 0 else 0
				# Gather status counts
				status_counts = {}
				for s in SequenceStatus:
					cnt = len(self.state.global_batch.get_sequences_by_status(s))
					if cnt > 0:
						status_counts[s.name] = cnt
				# Per-sequence host page stats
				host_pages_list = []
				for uuid in decode_uuids:
					seq = self.state.global_batch.get_sequence(uuid)
					if seq is not None:
						host_pages_list.append(seq.host_pages_allocated)
				chunk_val = self._worker._get_effective_chunk_size()
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
		watermark_triggered = self._kv.check_watermark_trigger()

		return decode_uuids, batch, new_async_task, new_load_uuids, new_load_local, new_load_global, timing, watermark_triggered
