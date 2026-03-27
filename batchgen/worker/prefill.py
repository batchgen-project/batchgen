"""
PrefillScheduler: Batch selection, configuration, and forward pass for prefill phase.

Extracted from batchgen_worker.py (Step 7 of scheduler split).
Handles: prefill batch selection, engine config, prefill forward pass (standard + prepacked).
"""
import logging
import math
import os
import time
from typing import List

import torch
import torch.distributed as dist
from tqdm import tqdm

from batchgen.worker.state import WorkerState
from batchgen.models.wrappers import AttnWrapperBase
from batchgen.utils import create_position_ids_from_attention_mask
from batchgen.prefill.prepack import prepack_sequences, get_prepack_stats
from batchgen.sequence import SequenceStatus, INITIAL_GPU_PAGE_BUFFER
from batchgen.batchgen_worker import query

# Attn_Wrapper is an alias for AttnWrapperBase
Attn_Wrapper = AttnWrapperBase

BATCHGEN_CB_DEBUG = os.environ.get("BATCHGEN_CB_LOG", "").upper() == "DEBUG"


class PrefillScheduler:
	"""Prefill batch selection, configuration, and forward pass.

	Uses a worker reference for shared methods (select_tokens, deep_free,
	set_phase, feed_watchdog) that are also used by the decode path.
	"""

	def __init__(self, state: WorkerState, index_manager, kv_manager, worker):
		self.state = state
		self._index = index_manager
		self._kv = kv_manager
		self._worker = worker  # for shared callbacks

	def prepare_batch(self) -> List[str]:
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
		if self.state.enable_host_kv_eviction:
			evicted_uuids = self.state.global_batch.get_sequences_by_status(SequenceStatus.EVICTED)
			# Weighted priority: more decoded tokens = higher priority (less wasted work)
			evicted_uuids.sort(key=lambda u: (
				-self.state.global_batch.get_sequence(u).total_decoded_before_eviction,
				self.state.global_batch.get_sequence(u).global_idx
			))

		queueing_uuids = self.state.global_batch.get_sequences_by_status(SequenceStatus.QUEUEING)
		queueing_uuids.sort(key=lambda uuid: self.state.global_batch.get_sequence(uuid).global_idx)

		all_candidates = evicted_uuids + queueing_uuids
		if not all_candidates:
			return []

		gpus_per_node = torch.cuda.device_count()
		num_nodes = self._worker._get_num_nodes()
		my_node = self._worker._get_node_for_rank(self.state.rank)
		chunk_size = self._worker._get_effective_chunk_size()

		# Step 1: Get this node's host KV free pages
		local_host_free = self._kv.get_host_free_pages()

		# Step 2: Gather host KV free pages from first rank on each node
		# Only rank 0, 8, 16, ... (first on each node) reports actual value
		if self.state.rank % gpus_per_node == 0:
			report_free = local_host_free
		else:
			report_free = 0  # Non-first ranks report 0

		free_tensor = torch.tensor([report_free], dtype=torch.int64, device=self.state.torch_device)
		gathered = [torch.zeros_like(free_tensor) for _ in range(self.state.world_size)]
		dist.all_gather(gathered, free_tensor)

		# Extract per-node host KV free pages
		per_node_host_free = []
		for node in range(num_nodes):
			first_rank = node * gpus_per_node
			per_node_host_free.append(int(gathered[first_rank].item()))

		if self.state.rank == 0:
			logging.info(f"Per-node host KV free pages: {per_node_host_free} (chunk_size={chunk_size})")

		# Step 3: Select sequences considering per-node host KV capacity
		# Use chunk-based pages instead of full kv_token_budget
		node_pages_used = [0] * num_nodes
		prefill_batch = []

		for uuid in all_candidates:
			seq = self.state.global_batch.get_sequence(uuid)
			assigned_rank = seq.assigned_rank
			seq_node = self._worker._get_node_for_rank(assigned_rank)

			# Dynamic reservation: only reserve initial chunk, not full budget
			req_pages = seq.get_host_pages_for_initial_chunk(chunk_size)

			if node_pages_used[seq_node] + req_pages <= per_node_host_free[seq_node]:
				prefill_batch.append(uuid)
				node_pages_used[seq_node] += req_pages

		if self.state.rank == 0:
			n_evicted = sum(1 for u in prefill_batch if self.state.global_batch.get_sequence(u).status == SequenceStatus.EVICTED)
			logging.info(
				f"[PREFILL] Selected {len(prefill_batch)} sequences "
				f"({n_evicted} recompute from eviction), "
				f"per-node pages: {node_pages_used}"
			)

		return prefill_batch

	def config_for_batch(self, prefill_uuids: List[str]) -> None:
		"""Configure prefill phase for a batch of sequences."""
		start_time = time.perf_counter()
		if self.state.rank == 0:
			logging.info(
				f"[PREFILL] Configuring prefill phase for {len(prefill_uuids)} sequences"
			)

		# DIAGNOSTIC: Log state of IN_DECODE/ON_HOLD sequences before prefill config
		in_decode = self.state.global_batch.get_sequences_by_status(SequenceStatus.IN_DECODE)
		on_hold = self.state.global_batch.get_sequences_by_status(SequenceStatus.ON_HOLD)
		prefilling = self.state.global_batch.get_sequences_by_status(SequenceStatus.PREFILLED)
		if (in_decode or on_hold) and BATCHGEN_CB_DEBUG:
			logging.debug(
				f"Rank {self.state.rank}: config_for_batch called while "
				f"{len(in_decode)} IN_DECODE, {len(on_hold)} ON_HOLD, {len(prefilling)} PREFILLED sequences exist. "
				f"This is a decode→prefill transition."
			)
			for uuid in (in_decode + on_hold)[:5]:
				seq = self.state.global_batch.get_sequence(uuid)
				logging.debug(
					f"Rank {self.state.rank}: Affected seq {seq.uuid[:8]}: "
					f"status={seq.status.name}, decoded_len={seq.decoded_length}, "
					f"ctx_len={seq.current_context_length}, gpu_pages={seq.gpu_pages_allocated}, "
					f"had_initial={seq.had_initial_gpu_reservation}"
				)

		# CRITICAL FIX: Flush pending KV append tasks before destroying GPU cache
		if self.state.pending_kv_tasks:
			logging.info(
				f"Rank {self.state.rank}: Flushing {len(self.state.pending_kv_tasks)} pending KV append tasks before prefill config"
			)
			self._kv.wait_pending_tasks()
			torch.cuda.synchronize(self.state.torch_device)

		# CRITICAL: Deep free decode model memory BEFORE configuring prefill (Bug Fix 7)
		logging.info("Deep freeing model memory before prefill config...")
		self._worker.deep_free_model_memory()

		# CRITICAL: Destroy GPU KV cache BEFORE configure_prefill (Bug Fix 7.2)
		self._kv.destroy_gpu_kv()

		if torch.cuda.is_available():
			free_mem, total_mem = torch.cuda.mem_get_info(self.state.local_rank)
			allocated = torch.cuda.memory_allocated(self.state.local_rank) / 1e9
			reserved = torch.cuda.memory_reserved(self.state.local_rank) / 1e9
			logging.info(
				f"[HBM] Rank {self.state.rank} BEFORE configure_prefill: "
				f"free={free_mem/1e9:.2f}GB alloc={allocated:.2f}GB rsv={reserved:.2f}GB"
			)

		# STEP 1: Configure model for prefill
		self.state.model, self._worker.weight_copy_task = self.state.parallel_manager.configure_prefill()
		self._worker.model = self.state.model  # Sync back
		self._worker.set_phase("prefill")

		if torch.cuda.is_available():
			torch.cuda.synchronize(self.state.torch_device)
			free_mem, total_mem = torch.cuda.mem_get_info(self.state.local_rank)
			allocated = torch.cuda.memory_allocated(self.state.local_rank) / 1e9
			logging.info(
				f"[HBM] Rank {self.state.rank} AFTER configure_prefill: "
				f"free={free_mem/1e9:.2f}GB alloc={allocated:.2f}GB"
			)

		self.state.core_engine.stop_h2d_worker()
		self.state.core_engine.clear_weight_copy_queue()
		self.state.core_engine.reset_prefill_buffer()
		self.state.core_engine.set_weight_copy_queue(self._worker.weight_copy_task)
		self.state.core_engine.start_h2d_worker()

		# STEP 3: Prepare evicted sequences for re-entry (before host KV allocation)
		for uuid in prefill_uuids:
			seq = self.state.global_batch.get_sequence(uuid)
			if seq.evicted_token_ids is None:
				continue

			evicted_ids = seq.evicted_token_ids  # 1D tensor
			new_prompt_len = len(evicted_ids)
			prev_decoded = seq.total_decoded_before_eviction

			# Rebuild input_ids with new prompt — reuse buffer pool slot
			seq_extended_size = seq.kv_token_budget
			slot = seq._buffer_slot
			self.state.buffer_pool.input_ids_buffer[slot, :] = 0
			self.state.buffer_pool.input_ids_buffer[slot, :new_prompt_len] = evicted_ids
			seq.input_ids = self.state.buffer_pool.get_input_ids_view(slot, seq_extended_size)

			seq.prompt_length = new_prompt_len
			seq.current_context_length = new_prompt_len

			# Pre-fill decoded_tokens with previously decoded tokens (Q1/Q2)
			self.state.buffer_pool.decoded_tokens_buffer[slot, :] = self.state.buffer_pool.pad_token_id
			seq.decoded_tokens = self.state.buffer_pool.get_decoded_tokens_view(slot)
			if prev_decoded > 0:
				old_decoded = evicted_ids[seq.original_prompt_length:]
				n_old = min(len(old_decoded), self.state.max_decoding_length)
				seq.decoded_tokens[0, :n_old] = old_decoded[:n_old]
				seq.decoded_length = n_old
			else:
				seq.decoded_length = 0

			# Remaining decode budget
			remaining_decode = max(0, seq.original_max_decode_length - prev_decoded)
			seq.max_decode_length = remaining_decode

			# Clear eviction state
			seq.evicted_token_ids = None

			# Recreate query_book entry for this rank's evicted sequences (Q4)
			if seq.assigned_rank == self.state.rank and uuid in self.state.uuid_to_local_map:
				local_idx = self.state.uuid_to_local_map[uuid]
				self._worker.query_book[local_idx] = query(
					text=seq.text,
					encoded={
						"input_ids": seq.input_ids,
					},
					decoded_tokens=seq.decoded_tokens,
					kv_token_budget=seq.kv_token_budget,
				)

			logging.info(
				f"Rank {self.state.rank}: Prepared EVICTED seq {uuid[:8]} for re-entry: "
				f"new_prompt={new_prompt_len}, prev_decoded={prev_decoded}, "
				f"remaining_decode={remaining_decode}, kv_budget={seq.kv_token_budget}"
			)

		# STEP 4: Allocate host KV pages for sequences (only THIS RANK's sequences)
		my_prefill_uuids = []
		for uuid in prefill_uuids:
			seq = self.state.global_batch.get_sequence(uuid)
			if seq.assigned_rank == self.state.rank:
				my_prefill_uuids.append(uuid)
				# Add to local maps if not already present (for new sequences)
				if uuid not in self.state.uuid_to_local_map:
					if self.state.free_local_indices:
						new_local_idx = self.state.free_local_indices.pop()
					else:
						new_local_idx = self.state.next_local_idx
						self.state.next_local_idx += 1
					self.state.uuid_to_local_map[uuid] = new_local_idx
					self.state.local_to_uuid_map[new_local_idx] = uuid
					logging.debug(
						f"Rank {self.state.rank}: Added new sequence {uuid[:8]}... to local maps "
						f"(local_idx={new_local_idx})"
					)

		if my_prefill_uuids:
			global_sequence_ids = []
			sequence_tokens = []
			chunk_size = self._worker._get_effective_chunk_size()

			for uuid in my_prefill_uuids:
				seq = self.state.global_batch.get_sequence(uuid)
				global_sequence_ids.append(seq.global_idx)
				post_prefill_length = seq.prompt_length + 1
				gpu_initial_pages = math.ceil(post_prefill_length / seq.PAGE_SIZE) + INITIAL_GPU_PAGE_BUFFER
				gpu_initial_tokens = gpu_initial_pages * seq.PAGE_SIZE
				initial_capacity = max(seq.prompt_length + chunk_size, gpu_initial_tokens)
				initial_capacity = min(initial_capacity, seq.kv_token_budget)
				sequence_tokens.append(initial_capacity)
				seq.host_token_capacity = initial_capacity
				seq.host_pages_allocated = math.ceil(initial_capacity / seq.PAGE_SIZE)

			logging.debug(
				f"Rank {self.state.rank}: Registering {len(global_sequence_ids)} sequences for host KV "
				f"(chunk_size={chunk_size})"
			)

			self.state.core_engine.host_paged_kv_worker_view.register_sequences(global_sequence_ids)
			self.state.core_engine.host_paged_kv_worker_view.allocate_pages_for_sequences(
				list(zip(global_sequence_ids, sequence_tokens))
			)

			kv_stats = self.state.core_engine.host_paged_kv_worker_view.get_stats()
			if self.state.rank == 0:
				logging.info(f"[PREFILL] Host KV allocated: {kv_stats.num_used_pages}/{kv_stats.num_total_pages} pages")

		if self.state.rank == 0:
			logging.info(f"[PREFILL] Config completed: {(time.perf_counter() - start_time)*1000:.1f}ms")

	def run_prefill(self, batch: list[int]):
		"""
		Handle the prefill for a batch.
		batch: list of local indices
		"""
		if "deepseek" in self.state.model_config.model_type:
			self.state.model.model._use_flash_attention_2 = False

		# Dynamic padding: find max length within THIS batch, not global max
		batch_seq_lengths = [
			self._worker.query_book[query_idx].encoded["input_ids"].shape[1]
			for query_idx in batch
		]
		batch_max_len = max(batch_seq_lengths)

		# Pad each sequence to batch_max_len and construct attention masks on-the-fly
		padded_input_ids = []
		padded_attention_masks = []
		for query_idx in batch:
			seq_input_ids = self._worker.query_book[query_idx].encoded["input_ids"]
			uuid = self.state.local_to_uuid_map[query_idx]
			seq = self.state.global_batch.get_sequence(uuid)
			prompt_len = seq.prompt_length
			seq_len = seq_input_ids.shape[1]

			# Construct attention mask from prompt_length
			seq_attention_mask = torch.zeros((1, seq_len), dtype=torch.int64)
			seq_attention_mask[0, :prompt_len] = 1

			if seq_len < batch_max_len:
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
			len(batch) / self._worker.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size
		)
		prefill_micro_batch_input_ids = torch.split(
			input_ids,
			self._worker.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size,
		)
		prefill_micro_batch_attention_masks = torch.split(
			attention_masks,
			self._worker.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size,
		)
		if self.state.rank == 0:
			logging.info(f"Number of prefill micro batches: {num_prefill_micro_batches}")

		cur_batch_start = 0
		output_tokens = []

		for micro_batch_idx in tqdm(range(num_prefill_micro_batches), desc="Prefill Micro Batch"):
			self._worker.feed_watchdog()

			with torch.inference_mode():
				Attn_Wrapper.attention_mask = prefill_micro_batch_attention_masks[micro_batch_idx]
				Attn_Wrapper.position_ids = create_position_ids_from_attention_mask(
					prefill_micro_batch_attention_masks[micro_batch_idx]
				)

				cur_batch_size = prefill_micro_batch_input_ids[micro_batch_idx].shape[0]
				cur_batch_local = batch[cur_batch_start : cur_batch_start + cur_batch_size]

				Attn_Wrapper.cur_batch = self._index.local_indices_to_global_seq_ids(cur_batch_local)

				cur_batch_start += cur_batch_size
				assert len(cur_batch_local) == cur_batch_size

				outputs = self.state.model(
					prefill_micro_batch_input_ids[micro_batch_idx].to(self.state.torch_device),
					attention_mask=prefill_micro_batch_attention_masks[micro_batch_idx].to(self.state.torch_device),
					use_cache=False,
				)
				new_tokens = self._worker._select_tokens(outputs.logits[:, -1, :])
				output_tokens.append(new_tokens)

		new_tokens = torch.cat(output_tokens, dim=0)

		# Update sequence state after prefill
		new_tokens_cpu = new_tokens.cpu()
		for i, local_idx in enumerate(batch):
			uuid = self.state.local_to_uuid_map[local_idx]
			seq = self.state.global_batch.get_sequence(uuid)
			token_pos = seq.decoded_length
			self._worker.query_book[local_idx].decoded_tokens[:, token_pos] = new_tokens_cpu[i]
			seq.decoded_length = token_pos + 1
			seq.current_context_length = seq.original_prompt_length + seq.decoded_length

			if self._worker._should_stop_at_eos(new_tokens_cpu[i].item()):
				seq.eos_reached = True

		return new_tokens

	def run_prefill_prepacked(self, batch: list[int]):
		"""
		Handle prefill for a batch using prepack optimization.

		Prepack combines multiple shorter sequences into rows to minimize padding waste,
		which is especially beneficial for MLP/MoE layers.

		Args:
			batch: list of local indices
		"""
		if "deepseek" in self.state.model_config.model_type:
			self.state.model.model._use_flash_attention_2 = False

		# Collect input_ids and attention_masks as lists for prepacking
		input_ids_list = []
		attention_mask_list = []
		seq_lengths = []

		for query_idx in batch:
			input_ids = self._worker.query_book[query_idx].encoded["input_ids"][:, :self._worker.max_input_length]
			uuid = self.state.local_to_uuid_map[query_idx]
			seq = self.state.global_batch.get_sequence(uuid)
			actual_len = min(seq.prompt_length, self._worker.max_input_length)
			seq_lengths.append(actual_len)

			# Construct attention mask on-the-fly from prompt_length
			attention_mask = torch.zeros_like(input_ids, dtype=torch.int64)
			attention_mask[0, :actual_len] = 1

			input_ids_list.append(input_ids)
			attention_mask_list.append(attention_mask)

		# Prepack sequences
		row_capacity = self._worker.engine_config.Module_Batching_Config.prepack_row_capacity
		prepack_meta = prepack_sequences(
			input_ids_list,
			attention_mask_list,
			row_capacity=row_capacity,
			device=self.state.torch_device,
		)

		# Log prepack statistics
		if self.state.rank == 0:
			stats = get_prepack_stats(prepack_meta)
			logging.info(
				f"Prepack stats: {stats['num_sequences']} seqs -> {stats['num_packed_rows']} rows, "
				f"padding saved: {stats['padding_saved']} tokens, "
				f"efficiency: {stats['packing_efficiency']:.2%}"
			)

		# Create flattened tensors for prepacked forward
		total_tokens = sum(prepack_meta.original_seq_lengths)

		packed_input_ids_flat = []
		packed_position_ids_flat = []

		for seq_idx in range(prepack_meta.num_original_sequences):
			row_idx, start_pos = prepack_meta.pack_assignment[seq_idx]
			seq_len = prepack_meta.original_seq_lengths[seq_idx]

			seq_input_ids = prepack_meta.packed_input_ids[row_idx, start_pos:start_pos + seq_len]
			packed_input_ids_flat.append(seq_input_ids)

			packed_position_ids_flat.append(torch.arange(seq_len, device=self.state.torch_device))

		packed_input_ids_flat = torch.cat(packed_input_ids_flat, dim=0)
		packed_position_ids_flat = torch.cat(packed_position_ids_flat, dim=0)

		# Split sequences into micro-batches based on TOKEN count (not sequence count)
		MAX_TOKENS_PER_MICRO_BATCH = self._worker.engine_config.Module_Batching_Config.prefill_micro_batch_token_cap
		num_sequences = prepack_meta.num_original_sequences
		seq_lengths_list = prepack_meta.original_seq_lengths

		micro_batches = []
		current_batch_start = 0
		current_batch_tokens = 0

		for seq_idx in range(num_sequences):
			seq_len = seq_lengths_list[seq_idx]

			if current_batch_tokens + seq_len > MAX_TOKENS_PER_MICRO_BATCH and current_batch_tokens > 0:
				micro_batches.append((current_batch_start, seq_idx))
				current_batch_start = seq_idx
				current_batch_tokens = 0

			current_batch_tokens += seq_len

		if current_batch_start < num_sequences:
			micro_batches.append((current_batch_start, num_sequences))

		if self.state.rank == 0:
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
				disable=(self.state.rank != 0)
			):
				self._worker.feed_watchdog()

				batch_seq_lengths = seq_lengths_list[seq_start:seq_end]
				batch_num_seqs = seq_end - seq_start

				batch_input_ids = []
				batch_position_ids = []
				token_offset = sum(seq_lengths_list[:seq_start])

				for seq_idx in range(seq_start, seq_end):
					seq_len = seq_lengths_list[seq_idx]
					seq_token_start = sum(seq_lengths_list[:seq_idx])
					seq_token_end = seq_token_start + seq_len

					batch_input_ids.append(packed_input_ids_flat[seq_token_start:seq_token_end])
					batch_position_ids.append(packed_position_ids_flat[seq_token_start:seq_token_end])

				batch_input_ids_flat = torch.cat(batch_input_ids, dim=0)
				batch_position_ids_flat = torch.cat(batch_position_ids, dim=0)

				batch_cu_seqlens = torch.zeros(batch_num_seqs + 1, dtype=torch.int32, device=self.state.torch_device)
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
				batch_local_indices = batch[seq_start:seq_end]
				Attn_Wrapper.cur_batch = self._index.local_indices_to_global_seq_ids(batch_local_indices)

				# CRITICAL: Also bind to AttnWrapperBase for models using new wrapper system (GPT-OSS)
				AttnWrapperBase.prepack_mode = True
				AttnWrapperBase.prepack_cu_seqlens = batch_cu_seqlens
				AttnWrapperBase.prepack_max_seqlen = batch_max_seqlen
				AttnWrapperBase.prepack_num_sequences = batch_num_seqs
				AttnWrapperBase.prepack_seq_lengths = batch_seq_lengths
				AttnWrapperBase.position_ids = batch_position_ids_flat
				AttnWrapperBase.cur_batch = Attn_Wrapper.cur_batch

				# Embed tokens
				inputs_embeds = self.state.model.model.embed_tokens(batch_input_ids_flat.to(self.state.torch_device))

				hidden_states = inputs_embeds.unsqueeze(0)

				# Forward through model layers
				for layer_idx, decoder_layer in enumerate(self.state.model.model.layers):
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
				hidden_states = self.state.model.model.norm(hidden_states)

				# Extract last token hidden states for each sequence
				last_token_indices = batch_cu_seqlens[1:] - 1
				last_token_hidden = hidden_states[0, last_token_indices, :]

				# DEBUG: Verify per-sequence hidden states after prefill
				if os.environ.get("BATCHGEN_DEBUG_PREFILL_OUTPUT", "0") == "1":
					print(f"\n[PREFILL OUTPUT DEBUG] === Micro-batch {batch_idx} ===")
					print(f"[PREFILL OUTPUT DEBUG] hidden_states.shape = {hidden_states.shape}")
					print(f"[PREFILL OUTPUT DEBUG] batch_cu_seqlens = {batch_cu_seqlens.tolist()}")
					print(f"[PREFILL OUTPUT DEBUG] last_token_indices = {last_token_indices.tolist()}")
					print(f"[PREFILL OUTPUT DEBUG] last_token_hidden.shape = {last_token_hidden.shape}")
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
					self.state.model.lm_head.weight,
					self.state.model.lm_head.bias if hasattr(self.state.model.lm_head, 'bias') and self.state.model.lm_head.bias is not None else None
				)

				batch_new_tokens = self._worker._select_tokens(logits)
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

		AttnWrapperBase.prepack_mode = False
		AttnWrapperBase.prepack_cu_seqlens = None
		AttnWrapperBase.prepack_max_seqlen = None
		AttnWrapperBase.prepack_num_sequences = None
		AttnWrapperBase.prepack_seq_lengths = None

		# Log timing summary
		self._worker._log_prefill_timing()

		new_tokens = torch.cat(output_tokens, dim=0)

		# Update sequence state after prefill
		new_tokens_cpu = new_tokens.cpu()
		for i, local_idx in enumerate(batch):
			uuid = self.state.local_to_uuid_map[local_idx]
			seq = self.state.global_batch.get_sequence(uuid)
			token_pos = seq.decoded_length
			self._worker.query_book[local_idx].decoded_tokens[:, token_pos] = new_tokens_cpu[i]
			seq.decoded_length = token_pos + 1
			seq.current_context_length = seq.original_prompt_length + seq.decoded_length

			if self._worker._should_stop_at_eos(new_tokens_cpu[i].item()):
				seq.eos_reached = True

		return new_tokens
