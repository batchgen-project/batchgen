"""
HostKVRebalancer: Host KV migration, eviction, and on-hold management.

Extracted from batchgen_worker.py (Step 10 of scheduler split).
Handles: cross-node KV migration, sequence eviction, on-hold transitions.
"""
import logging
import math
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.distributed as dist

from batchgen.worker.state import WorkerState
from batchgen.migration import MigrationOp, HostKVStats
from batchgen.sequence import SequenceStatus
from batchgen.batchgen_worker import query

NUM_GPUS_PER_NODE = int(os.environ.get('NUM_GPUS_PER_NODE', '8'))
BATCHGEN_CB_DEBUG = os.environ.get("BATCHGEN_CB_LOG", "").upper() == "DEBUG"
BATCHGEN_ENABLE_CRITICAL_DIAGS = os.environ.get('BATCHGEN_ENABLE_CRITICAL_DIAGS', '0') == '1'


class HostKVRebalancer:
	"""Host KV migration, eviction, and on-hold management.

	Uses worker reference for shared methods and GPU KV manager access.
	"""

	def __init__(self, state: WorkerState, index_manager, kv_manager, sync_coordinator, worker):
		self.state = state
		self._index = index_manager
		self._kv = kv_manager
		self._sync = sync_coordinator
		self._worker = worker
		self._gloo_group = None
		self._dest_rank_counter = defaultdict(int)

	def select_for_onhold(
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
		manager = self._worker.gpu_paged_kv_cache_manager
		current_free = manager.get_stats().num_free_pages if manager else 0
		pages_to_free = required_free_pages - current_free

		if pages_to_free <= 0:
			return []

		# Sort by decoded_length ASCENDING (least progress first - evict these)
		candidates = []
		for uuid in active_uuids:
			if uuid not in self.state.uuid_to_local_map:
				continue
			seq = self.state.global_batch.get_sequence(uuid)
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

	def put_onhold(self, uuids: List[str]) -> None:
		"""Put sequences ON_HOLD: release GPU KV pages, keep host KV."""
		if not uuids:
			return
		
		my_uuids = [u for u in uuids if u in self.state.uuid_to_local_map]
		
		if my_uuids:
			local_indices = self._index.get_local_indices_for_uuids(my_uuids)
			global_ids = self._index.local_indices_to_global_seq_ids(local_indices)
			
			manager = self._worker.gpu_paged_kv_cache_manager
			if manager is not None:
				manager.free_pages_for_sequences(global_ids)
			
			for uuid in my_uuids:
				seq = self.state.global_batch.get_sequence(uuid)
				seq.gpu_pages_allocated = 0
				self.state.sequences_with_gpu_kv.discard(uuid)
		
		# self._worker._update_batch_status(uuids, SequenceStatus.ON_HOLD)
		
		# FIX: Rebuild page table with remaining active sequences
		manager = self._worker.gpu_paged_kv_cache_manager
		if manager is not None and manager.is_initialized:
			remaining_uuids = [u for u in self.state.sequences_with_gpu_kv if u not in set(uuids)]
			if remaining_uuids:
				remaining_local = self._index.get_local_indices_for_uuids(remaining_uuids)
				remaining_global = self._index.local_indices_to_global_seq_ids(remaining_local)
				manager.rebuild_page_table(remaining_global)


	def _get_or_create_gloo_group(self):
		"""Get or create a Gloo process group for CPU tensor migrations.

		Gloo backend supports CPU tensors and can use RDMA if available.
		This is more memory efficient than NCCL (which requires GPU staging).

		Returns:
			The Gloo process group for CPU tensor operations.
		"""
		if not hasattr(self, '_gloo_migration_group') or self._gloo_migration_group is None:
			logging.debug(f"Rank {self.state.rank}: Creating Gloo process group for CPU migrations")
			# Create a new group with Gloo backend including all ranks
			self._gloo_migration_group = dist.new_group(
				ranks=list(range(self.state.world_size)),
				backend="gloo"
			)
			logging.debug(f"Rank {self.state.rank}: Gloo process group created")
		return self._gloo_migration_group

	def _destroy_gloo_group(self):
		"""Destroy the Gloo process group after migrations are done."""
		if hasattr(self, '_gloo_migration_group') and self._gloo_migration_group is not None:
			logging.debug(f"Rank {self.state.rank}: Destroying Gloo process group")
			dist.destroy_process_group(self._gloo_migration_group)
			self._gloo_migration_group = None

	def plan_migration(self) -> List[MigrationOp]:
		"""Plan sequence migrations to rebalance host KV across nodes.

		Returns:
			List of MigrationOp objects describing planned migrations.
		"""
		# Gather host KV stats from all local_rank 0
		if self.state.local_rank == 0:
			local_stats = self._kv.get_host_utilization()
		else:
			local_stats = None

		all_stats = [None] * self.state.world_size
		dist.all_gather_object(all_stats, local_stats)
		node_stats = {s['node_id']: s for s in all_stats if s is not None}

		if len(node_stats) <= 1:
			# Only one node, no migration needed
			if self.state.rank == 0:
				logging.info("MIGRATION: Single node detected, skipping rebalancing")
			return []

		# Calculate target pages per node
		total_used = sum(s['num_used_pages'] for s in node_stats.values())
		num_nodes = len(node_stats)
		target_per_node = total_used // num_nodes

		if self.state.rank == 0:
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
			if self.state.rank == 0:
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
					if src_rank >= self.state.world_size:
						break
					for status in [SequenceStatus.PREFILLED, SequenceStatus.ON_HOLD]:
						for uuid in self.state.global_batch.get_sequences_for_rank_with_status(src_rank, status):
							if uuid not in migrated_uuids:
								candidate_sequences.append(uuid)

				if not candidate_sequences:
					if self.state.rank == 0:
						if BATCHGEN_CB_DEBUG:
							logging.debug(f"MIGRATION: No more candidates on node {src_node_id}, stopping")
					break

				# CRITICAL: Sort candidates deterministically before selection
				# Set operations (get_sequences_for_rank_with_status) don't preserve order,
				# so we must sort to ensure all ranks pick the same sequence
				candidate_sequences.sort(key=lambda u: self.state.global_batch.get_sequence(u).global_idx)

				# Pick smallest sequence (better packing), with global_idx as tie-breaker
				# This ensures deterministic selection across all ranks
				uuid = min(candidate_sequences, key=lambda u: (
					self.state.global_batch.get_sequence(u).kv_token_budget,
					self.state.global_batch.get_sequence(u).global_idx  # Tie-breaker
				))
				seq = self.state.global_batch.get_sequence(uuid)
				# CRITICAL FIX: Use actual host pages allocated, not full kv_token_budget.
				# Host KV uses chunked growth, so host_pages_allocated < ceil(kv_token_budget/PAGE_SIZE).
				# Using kv_token_budget causes IndexError when loading more pages than host has.
				pages_needed = seq.host_pages_allocated
				if pages_needed <= 0:
					if self.state.rank == 0:
						logging.warning(
							f"MIGRATION: Skipping seq {uuid[:8]}... - no host pages allocated"
						)
					migrated_uuids.add(uuid)  # Don't retry
					continue

				if self.state.rank == 0:
					if BATCHGEN_CB_DEBUG:
						logging.debug(
							f"MIGRATION: Selected seq {uuid[:8]}... from {len(candidate_sequences)} candidates "
							f"(global_idx={seq.global_idx}, from_rank={seq.assigned_rank}, "
							f"host_pages={pages_needed}, budget_pages={math.ceil(seq.kv_token_budget / self._worker.PAGE_SIZE)})"
						)

				# Find dest node with most free space (lowest used pages)
				# Use node_id as tie-breaker for determinism
				dest_node_id = min(underutilized, key=lambda x: (used_by_node[x[0]], x[0]))[0]

				# Check dest node has enough free pages for this migration
				dest_total = node_stats[dest_node_id]['num_total_pages']
				dest_free = dest_total - used_by_node[dest_node_id]
				if pages_needed > dest_free:
					if self.state.rank == 0:
						logging.info(
							f"MIGRATION: Dest node {dest_node_id} has insufficient free pages "
							f"({dest_free} free, need {pages_needed}), removing from candidates"
						)
					underutilized = [(nid, s) for nid, s in underutilized if nid != dest_node_id]
					if not underutilized:
						break
					continue

				# Distribute across ranks on dest node for load balancing
				# Use round-robin based on migration count to this node
				# (counter is reset at start of each planning round)
				if dest_node_id not in self._dest_rank_counter:
					self._dest_rank_counter[dest_node_id] = 0

				dest_rank_offset = self._dest_rank_counter[dest_node_id] % NUM_GPUS_PER_NODE
				dest_rank = dest_node_id * NUM_GPUS_PER_NODE + dest_rank_offset
				if dest_rank >= self.state.world_size:
					dest_rank = dest_node_id * NUM_GPUS_PER_NODE  # Fallback to rank 0
				self._dest_rank_counter[dest_node_id] += 1

				# Record migration using MigrationOp dataclass
				migrations.append(MigrationOp(
					uuid=uuid,
					from_rank=seq.assigned_rank,
					to_rank=dest_rank,
					pages=pages_needed,
					host_pages=pages_needed,
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
			if self.state.rank == 0:
				logging.warning(f"MIGRATION: Removed duplicates, {len(migrations)} unique migrations remain")

		if self.state.rank == 0:
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

	def execute_migrations(self, migrations: List[MigrationOp]) -> None:
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
		rounds = self._group_migrations(migrations)

		if self.state.rank == 0:
			logging.info(f"MIGRATION: Executing {len(migrations)} migrations in {len(rounds)} parallel rounds")

		for round_idx, round_migrations in enumerate(rounds):
			if self.state.rank == 0:
				logging.info(f"MIGRATION: Round {round_idx+1}/{len(rounds)}: {len(round_migrations)} parallel migrations")

			# Find if this rank participates in this round
			my_migration = None
			for mig in round_migrations:
				if self.state.rank == mig.from_rank or self.state.rank == mig.to_rank:
					my_migration = mig
					break

			# Execute migration if participating, otherwise just sync tensor shape info
			if my_migration is not None:
				# Verify sequence exists and is in expected state before migration
				seq = self.state.global_batch.get_sequence(my_migration.uuid)
				if seq is None:
					logging.error(f"MIGRATION: Rank {self.state.rank}: SKIP migration - seq {my_migration.uuid[:8]}... not found!")
				else:
					if BATCHGEN_CB_DEBUG:
						logging.debug(
							f"MIGRATION: Rank {self.state.rank}: Executing migration for {my_migration.uuid[:8]}... "
							f"(global_idx={seq.global_idx}, status={seq.status}, assigned_rank={seq.assigned_rank})"
						)
					self._execute_single_migration(
						uuid=my_migration.uuid,
						from_rank=my_migration.from_rank,
						to_rank=my_migration.to_rank
					)

			# Barrier after each round to ensure all transfers in this round complete
			dist.barrier()

		if self.state.rank == 0:
			logging.info(f"MIGRATION: All {len(rounds)} parallel rounds completed")

	def _group_migrations(self, migrations: List[MigrationOp]) -> List[List[MigrationOp]]:
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

	def _execute_single_migration(self, uuid: str, from_rank: int, to_rank: int) -> None:
		"""Migrate KV cache for one sequence from source to dest rank.

		Migration path: Direct host-to-host copy via network (no GPU staging)
		Uses PyTorch distributed send/recv on CPU tensors for efficient inter-node transfer.

		Args:
			uuid: Sequence UUID to migrate
			from_rank: Source rank (current owner)
			to_rank: Destination rank (new owner)
		"""
		seq = self.state.global_batch.get_sequence(uuid)
		if seq is None:
			logging.error(f"Rank {self.state.rank}: Cannot migrate {uuid[:8]}... - sequence not found")
			return

		# CRITICAL: Use GPU KV manager's config for tensor shape - this matches what
		# copy_kv_to_tensor() returns and what copy_tensor_to_kv() expects.
		# For MLA: num_k_heads=1 (latent attention), k_head_dim=576 (compressed KV)
		# Do NOT use model_config.num_key_value_heads or loaded_model_config.qk_rope_head_dim
		# as those have different values!
		gpu_kv_config = self._worker.gpu_paged_kv_cache_manager.config
		num_layers = self.state.model_config.num_hidden_layers
		num_k_heads = gpu_kv_config.num_k_heads  # For MLA: 1
		k_head_dim = gpu_kv_config.k_head_dim    # For MLA: 576 (compressed KV)
		kv_dtype = gpu_kv_config.kv_dtype
		page_size = gpu_kv_config.page_size_tokens  # Should be 64

		global_idx = seq.global_idx
		# CRITICAL FIX: Use actual host pages allocated, not full kv_token_budget.
		# Host KV uses chunked growth, so host_pages_allocated < ceil(kv_token_budget/page_size).
		# Using kv_token_budget causes IndexError when loading more pages than host has.
		pages_needed = seq.host_pages_allocated
		if pages_needed <= 0:
			logging.error(f"Rank {self.state.rank}: Cannot migrate {uuid[:8]}... - no host pages allocated")
			return

		# Tensor shape matches GPU KV manager: [num_layers, pages, page_size, num_k_heads, k_head_dim]
		k_shape = (num_layers, pages_needed, page_size, num_k_heads, k_head_dim)
		if self.state.rank == from_rank:
			# ===== SOURCE RANK: Read from host KV, send directly over network =====
			t0 = time.perf_counter()
			logging.debug(
				f"[MIGRATION] Rank {self.state.rank}: Send {uuid[:8]}... → rank {to_rank} "
				f"({pages_needed} pages)"
			)
			# Allocate CPU buffer for KV data
			# We'll load host KV → GPU → CPU buffer, then send
			# (Temporary workaround - ideally would read directly from host memory)
			manager = self._worker.gpu_paged_kv_cache_manager
			worker_view = self.state.host_kv_view
			if manager is None:
				logging.error(f"Rank {self.state.rank}: GPU KV manager not initialized")
				return
			# Ensure GPU KV manager is initialized (may be destroyed between decode/prefill phases)
			if not manager.is_initialized:
				logging.debug(f"[MIGRATION] Rank {self.state.rank}: Re-initializing GPU KV manager for migration")
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
					f"MIGRATION: Rank {self.state.rank}: Loading host KV for {uuid[:8]}... "
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
			torch.cuda.synchronize(self.state.torch_device)
			t_load = time.perf_counter()
			if BATCHGEN_CB_DEBUG:
				logging.debug(f"MIGRATION: Rank {self.state.rank}: Host→GPU load: {(t_load-t0)*1000:.1f}ms")
			# Extract to contiguous tensor on GPU
			k_gpu = manager.copy_kv_to_tensor(global_idx)
			t_extract = time.perf_counter()
			if BATCHGEN_CB_DEBUG:
				logging.debug(f"MIGRATION: Rank {self.state.rank}: GPU tensor extraction: {(t_extract-t_load)*1000:.1f}ms")

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
					f"MIGRATION SEND: Rank {self.state.rank}: Validating KV for {uuid[:8]}... (global_idx={global_idx}): "
					f"k_gpu_shape={list(k_gpu.shape)}, valid_tokens={valid_tokens}/{total_slots}, "
					f"layer0_mean={send_k_mean:.4f}, std={send_k_std:.4f}, "
					f"has_nan={send_has_nan}, is_zero={send_is_zero}{padding_info}, "
					f"first_values={valid_k[0, 0, :4].tolist() if valid_k.numel() > 0 else 'N/A'}"
				)
				if send_is_zero:
					logging.error(
						f"MIGRATION SEND: Rank {self.state.rank}: CRITICAL - KV to send is ALL ZEROS for {uuid[:8]}! "
						f"Host KV may be corrupted or load failed."
					)
				if send_has_nan:
					logging.error(f"MIGRATION SEND: Rank {self.state.rank}: CRITICAL - KV to send has NaN for {uuid[:8]}!")
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
							f"MIGRATION SEND: Rank {self.state.rank}: NaN layer analysis for {uuid[:8]}: "
							f"total_nan_layers={len(nan_layers)}/{k_gpu.shape[0]}, "
							f"details={nan_layers[:5]}"  # First 5 layers with NaN
						)

			# Move GPU → CPU for Gloo transfer (Gloo supports CPU tensors, more memory efficient)
			k_cpu = k_gpu.cpu().contiguous()
			t_cpu = time.perf_counter()
			if BATCHGEN_CB_DEBUG:
				logging.debug(f"MIGRATION: Rank {self.state.rank}: GPU→CPU copy: {(t_cpu-t_extract)*1000:.1f}ms")
			# Send via Gloo backend (supports CPU tensors and RDMA if available)
			gloo_group = self._get_or_create_gloo_group()
			dist.send(tensor=k_cpu, dst=to_rank, group=gloo_group)
			t_send = time.perf_counter()
			if BATCHGEN_CB_DEBUG:
				logging.debug(f"MIGRATION: Rank {self.state.rank}: Gloo send: {(t_send-t_cpu)*1000:.1f}ms")
			# Free GPU pages
			manager.free_pages_for_sequences([global_idx])
			# Free host KV pages
			worker_view.release_sequence_pages([global_idx])
			# Also send query_book data (input_ids, decoded_tokens)
			local_idx = self.state.uuid_to_local_map.get(uuid)
			if local_idx is not None and local_idx in self._worker.query_book:
				qb = self._worker.query_book[local_idx]
				# Send tensors via Gloo — must use .clone() because buffer pool views
				# are already contiguous (.contiguous() returns same tensor, not a copy)
				dist.send(tensor=qb.encoded["input_ids"].clone(), dst=to_rank, group=gloo_group)
				dist.send(tensor=qb.decoded_tokens.clone(), dst=to_rank, group=gloo_group)
				# Free buffer slot after send completes
				seq_for_slot = self.state.global_batch.get_sequence(uuid)
				if hasattr(seq_for_slot, '_buffer_slot') and seq_for_slot._buffer_slot >= 0:
					self.state.buffer_pool.free_slot(seq_for_slot._buffer_slot)
					seq_for_slot._buffer_slot = -1
				if BATCHGEN_CB_DEBUG:
					logging.debug(f"MIGRATION: Rank {self.state.rank}: Sent query_book for {uuid[:8]}...")
			else:
				logging.warning(f"MIGRATION: Rank {self.state.rank}: No query_book entry for {uuid[:8]}... (local_idx={local_idx})")

			t_total = time.perf_counter()
			if BATCHGEN_CB_DEBUG:
				logging.debug(
					f"MIGRATION: Rank {self.state.rank}: Sent {uuid[:8]}... "
					f"in {(t_total-t0)*1000:.1f}ms"
				)
		elif self.state.rank == to_rank:
			# ===== DEST RANK: Receive over network via Gloo, write to host KV =====
			t0 = time.perf_counter()
			if BATCHGEN_CB_DEBUG:
				logging.debug(
					f"MIGRATION: Rank {self.state.rank}: Recv {uuid[:8]}... ← rank {from_rank} "
					f"({pages_needed} pages)"
				)
			# Allocate CPU buffer for receiving (Gloo supports CPU tensors)
			k_cpu = torch.empty(k_shape, dtype=kv_dtype, device="cpu", pin_memory=True)
			# Receive via Gloo backend
			gloo_group = self._get_or_create_gloo_group()
			dist.recv(tensor=k_cpu, src=from_rank, group=gloo_group)
			t_recv = time.perf_counter()
			if BATCHGEN_CB_DEBUG:
				logging.debug(f"MIGRATION: Rank {self.state.rank}: Gloo recv: {(t_recv-t0)*1000:.1f}ms")
			# Register and allocate host KV pages
			worker_view = self.state.host_kv_view
			tokens_needed = pages_needed * page_size
			worker_view.register_sequences([global_idx])
			worker_view.allocate_pages_for_sequences([(global_idx, tokens_needed)])
			t_alloc = time.perf_counter()
			if BATCHGEN_CB_DEBUG:
				logging.debug(f"MIGRATION: Rank {self.state.rank}: Host allocation: {(t_alloc-t_recv)*1000:.1f}ms")
			# Move CPU → GPU for offload to host KV
			k_gpu = k_cpu.to(self.device, non_blocking=True)
			torch.cuda.synchronize(self.state.torch_device)
			t_gpu = time.perf_counter()
			if BATCHGEN_CB_DEBUG:
				logging.debug(f"MIGRATION: Rank {self.state.rank}: CPU→GPU copy: {(t_gpu-t_alloc)*1000:.1f}ms")

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
						f"MIGRATION: Rank {self.state.rank}: Validating received KV for {uuid[:8]}... (global_idx={global_idx}): "
						f"k_gpu_shape={list(k_gpu.shape)}, valid_tokens={valid_tokens}/{total_slots}, "
						f"layer0_mean={migration_k_mean:.4f}, std={migration_k_std:.4f}, "
						f"has_nan={migration_has_nan}, is_zero={migration_is_zero}{padding_info}, "
						f"first_values={valid_k[0, 0, :4].tolist() if valid_k.numel() > 0 else 'N/A'}"
					)
				if migration_is_zero:
					logging.error(
						f"MIGRATION RECV: Rank {self.state.rank}: CRITICAL - Received KV is ALL ZEROS for {uuid[:8]}! "
						f"This means network transfer failed or source had invalid data."
					)
				if migration_has_nan:
					logging.error(
						f"MIGRATION RECV: Rank {self.state.rank}: CRITICAL - Received KV has NaN for {uuid[:8]}!"
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
							f"MIGRATION RECV: Rank {self.state.rank}: NaN layer analysis for {uuid[:8]}: "
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
			torch.cuda.synchronize(self.state.torch_device)
			# Clear held references for migration offload tensors so memory
			# can be reclaimed now that copies are guaranteed complete.
			if hasattr(self, '_pending_migration_offload_tensors'):
				self._pending_migration_offload_tensors.clear()
			t_store = time.perf_counter()
			if BATCHGEN_CB_DEBUG:
				logging.debug(f"MIGRATION: Rank {self.state.rank}: GPU→Host offload all layers: {(t_store-t_gpu)*1000:.1f}ms")
			# Note: GPU tensor k_gpu was only a staging buffer, not allocated in GPU paged KV manager
			# It will be freed automatically when it goes out of scope

			# Receive query_book data (input_ids, decoded_tokens)
			# Get tensor shapes from the Sequence object (all ranks have seq metadata)
			input_ids_shape = seq.input_ids.shape
			decoded_tokens_shape = seq.decoded_tokens.shape

			input_ids_recv = torch.empty(input_ids_shape, dtype=seq.input_ids.dtype, device="cpu")
			decoded_tokens_recv = torch.empty(decoded_tokens_shape, dtype=seq.decoded_tokens.dtype, device="cpu")

			dist.recv(tensor=input_ids_recv, src=from_rank, group=gloo_group)
			dist.recv(tensor=decoded_tokens_recv, src=from_rank, group=gloo_group)

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
				'decoded_tokens': decoded_tokens_recv,
				'kv_token_budget': seq.kv_token_budget,
			}
			if BATCHGEN_CB_DEBUG:
				logging.debug(f"MIGRATION: Rank {self.state.rank}: Recvd query_book for {uuid[:8]}...")

			t_total = time.perf_counter()
			if BATCHGEN_CB_DEBUG:
				logging.debug(
					f"MIGRATION: Rank {self.state.rank}: Recvd {uuid[:8]}... "
					f"in {(t_total-t0)*1000:.1f}ms"
				)
		# No barrier here - will be done in _rebalance_host_kv after all migrations

	def rebalance(self) -> None:
		"""Rebalance host KV cache by migrating sequences between nodes.

		Called during _config_prefill_for_batch() before assigning new sequences.
		This orchestrates the full rebalancing process:
		1. Plan migrations (deterministic across all ranks)
		2. Execute all migrations (NCCL transfers)
		3. Barrier to ensure all transfers complete
		4. Update sequence ownership metadata
		5. Barrier to ensure metadata consistency
		"""
		if not self.state.enable_decode_preemption:
			return

		rebalance_start = time.perf_counter()
		if self.state.rank == 0:
			logging.info("REBALANCE: Starting host KV rebalancing")

		# Plan migrations (all ranks compute same plan deterministically)
		migrations = self.plan_migration()

		if not migrations:
			if self.state.rank == 0:
				logging.info("REBALANCE: No migrations needed, host KV already balanced")
			return

		# Log migration summary
		if self.state.rank == 0:
			total_pages = sum(m.pages for m in migrations)
			logging.info(
				f"REBALANCE: Executing {len(migrations)} migrations "
				f"({total_pages} total pages, ~{total_pages * 64} tokens)"
			)

		# STEP 1: Execute all migrations in parallel (host-to-host transfers)
		# Parallel execution utilizes all network cards by having multiple rank pairs
		# communicate simultaneously
		migration_start = time.perf_counter()
		self.execute_migrations(migrations)
		migration_end = time.perf_counter()
		if self.state.rank == 0:
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
				self.state.global_batch.assign_rank(uuid, new_rank)
			except KeyError:
				logging.error(f"Rank {self.state.rank}: Cannot update ownership for {uuid[:8]}... - sequence not found")
				continue

			# IMPORTANT: Don't change sequence status - it remains PREFILLED or ON_HOLD
			# The sequence is still valid, just owned by a different rank now

			# Update host KV tracking to match actual allocation on dest.
			# All ranks execute this (migration list is deterministic), keeping fields consistent.
			seq = self.state.global_batch.get_sequence(uuid)
			if seq is not None:
				seq.host_pages_allocated = mig.host_pages
				seq.host_token_capacity = mig.host_pages * self._worker.PAGE_SIZE

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
			self._sync.sync_metadata(migrated_uuids)
			logging.info(
				f"Rank {self.state.rank}: REBALANCE: Synced metadata for {len(migrated_uuids)} sequences "
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
			if self.state.rank == old_rank:
				local_idx = self.state.uuid_to_local_map.pop(uuid, None)
				if local_idx is not None:
					self.state.local_to_uuid_map.pop(local_idx, None)
					self.state.sequences_with_gpu_kv.discard(uuid)
					# Remove query_book entry
					self._worker.query_book.pop(local_idx, None)
					# Add freed index to free list for O(1) reuse
					self.state.free_local_indices.add(local_idx)
					logging.debug(f"Rank {self.state.rank}: Removed {uuid[:8]}... from local mappings (freed local_idx={local_idx})")

			# Update local mappings on dest rank (add)
			if self.state.rank == new_rank:
				# O(1) allocation: prefer reusing freed indices, otherwise use next available
				if self.state.free_local_indices:
					new_local_idx = self.state.free_local_indices.pop()
				else:
					new_local_idx = self.state.next_local_idx
					self.state.next_local_idx += 1

				self.state.uuid_to_local_map[uuid] = new_local_idx
				self.state.local_to_uuid_map[new_local_idx] = uuid
				# Note: Don't add to _sequences_with_gpu_kv - KV is in host, not GPU

				# Create query_book entry from pending migrated data, copying into buffer pool
				if hasattr(self, '_pending_migrated_query_book') and uuid in self._pending_migrated_query_book:
					pending = self._pending_migrated_query_book.pop(uuid)
					budget = pending['kv_token_budget']
					# Reuse existing buffer slot — Phase 3 already allocated a slot for every
					# sequence in global_batch, so seq._buffer_slot is valid
					seq = self.state.global_batch.get_sequence(uuid)
					existing_slot = seq._buffer_slot
					logging.info(
						f"Rank {self.state.rank}: Migration receive {uuid[:8]}: "
						f"reusing existing_slot={existing_slot}, budget={budget}"
					)
					if existing_slot < 0:
						logging.error(f"Rank {self.state.rank}: Migration receive {uuid[:8]} has no buffer slot, allocating new")
						existing_slot = self.state.buffer_pool.allocate_slot()
						seq._buffer_slot = existing_slot
					self.state.buffer_pool.input_ids_buffer[existing_slot, :budget] = pending['input_ids'][0, :budget]
					self.state.buffer_pool.decoded_tokens_buffer[existing_slot, :] = pending['decoded_tokens'][0, :]
					input_ids_view = self.state.buffer_pool.get_input_ids_view(existing_slot, budget)
					decoded_view = self.state.buffer_pool.get_decoded_tokens_view(existing_slot)
					seq.input_ids = input_ids_view
					seq.decoded_tokens = decoded_view
					self._worker.query_book[new_local_idx] = query(
						text=pending['text'],
						encoded={"input_ids": input_ids_view},
						decoded_tokens=decoded_view,
						kv_token_budget=budget,
					)
					logging.debug(f"Rank {self.state.rank}: Created query_book[{new_local_idx}] for migrated {uuid[:8]}...")
				else:
					logging.error(f"Rank {self.state.rank}: No pending query_book data for migrated {uuid[:8]}...")

				logging.debug(f"Rank {self.state.rank}: Added {uuid[:8]}... to local mappings (new local_idx={new_local_idx})")

		# BARRIER 2: Ensure all local mapping updates are complete across all ranks
		dist.barrier()

		# NOTE: Metadata sync was already done BEFORE local mapping updates (above)
		# At this point, all ranks have consistent metadata for migrated sequences.

		rebalance_end = time.perf_counter()
		if self.state.rank == 0:
			logging.info(
				f"[REBALANCE] Completed: {len(migrations)} sequences migrated "
				f"in {(rebalance_end-rebalance_start)*1000:.1f}ms total"
			)

			# Log final distribution
			if self.state.local_rank == 0:
				final_stats = self._kv.get_host_utilization()
				logging.info(
					f"  Node {final_stats['node_id']} final state: "
					f"{final_stats['num_used_pages']}/{final_stats['num_total_pages']} pages "
					f"({100-final_stats['free_percent']}% utilized)"
				)


	def put_on_hold(self, uuids: List[str]) -> None:
		"""Move IN_DECODE sequences to ON_HOLD, freeing GPU KV but keeping host KV."""
		if not uuids:
			return

		if self.state.rank == 0:
			logging.info(
				f"[WATERMARK] Putting {len(uuids)} sequences ON_HOLD"
			)

		# CRITICAL FIX: Sync sequence metadata BEFORE putting on hold
		# This ensures all ranks have consistent current_context_length values
		# which is essential for correct KV migration validation later
		self._sync.sync_metadata(uuids)

		# Free GPU pages for these sequences
		# CRITICAL FIX: GPU KV manager uses global_idx (not local_idx) as sequence ID
		if hasattr(self, 'gpu_paged_kv_cache_manager') and self._worker.gpu_paged_kv_cache_manager:
			global_seq_ids = []
			for uuid in uuids:
				seq = self.state.global_batch.get_sequence(uuid)
				if seq.assigned_rank == self.state.rank:
					# Verify sequence is in local map (should be for IN_DECODE sequences)
					if uuid in self.state.uuid_to_local_map:
						global_seq_ids.append(seq.global_idx)  # Use global_idx, not local_idx!

			if global_seq_ids:
				# Filter to only sequences the GPU manager actually tracks
				mgr = self._worker.gpu_paged_kv_cache_manager
				known_ids = [gid for gid in global_seq_ids if gid in mgr._sequences]
				if known_ids:
					mgr.free_pages_for_sequences(known_ids)
				if len(known_ids) < len(global_seq_ids):
					unknown = len(global_seq_ids) - len(known_ids)
					logging.debug(
						f"Rank {self.state.rank}: Skipped freeing {unknown} sequences not in GPU KV manager"
					)
				# Also remove from tracking set
				for uuid in uuids:
					seq = self.state.global_batch.get_sequence(uuid)
					if seq.assigned_rank == self.state.rank:
						self.state.sequences_with_gpu_kv.discard(uuid)

		# Update sequence status and reset GPU allocation
		# NOTE: Only reset gpu_pages_allocated, NOT had_initial_gpu_reservation.
		# ON_HOLD sequences are continuing decode when reloaded, so they should
		# get EXTENSION_GPU_PAGE_BUFFER (smaller), not INITIAL_GPU_PAGE_BUFFER.
		for uuid in uuids:
			seq = self.state.global_batch.get_sequence(uuid)
			seq.gpu_pages_allocated = 0
			self.state.global_batch.update_status(uuid, SequenceStatus.ON_HOLD)

		# Synchronize state across all ranks
		dist.barrier()
