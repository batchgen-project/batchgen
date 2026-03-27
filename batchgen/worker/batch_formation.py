"""
BatchFormation: Tokenization, rank assignment, and query book construction.

Extracted from batchgen_worker.py (Step 6 of scheduler split).
Handles the setup phase: tokenize → assign to ranks → build local query book.
"""
import logging
import os
import time
from typing import Dict, List, Optional, Set

import torch
import torch.distributed as dist

from batchgen.worker.state import WorkerState
from batchgen.batchgen_worker import query, QueryBookBufferPool


class BatchFormation:
	"""Tokenization, rank assignment, and query book construction.

	Called sequentially during process_new_batch():
	  tokenize() → assign_to_ranks() → build_query_book()
	"""

	def __init__(self, state: WorkerState):
		self.state = state

	def tokenize(self) -> None:
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
		if self.state.global_batch is None:
			raise RuntimeError("Global batch not initialized")

		# Phase 1: PARALLEL batch tokenization across ranks
		# Each rank tokenizes sequences[rank::world_size] to divide the work
		all_texts = [seq.text for seq in self.state.global_batch]
		num_sequences = len(all_texts)

		# Determine this rank's subset of sequences to tokenize
		my_indices = list(range(self.state.rank, num_sequences, self.state.world_size))
		my_texts = [all_texts[i] for i in my_indices]

		if self.state.rank == 0:
			logging.info(
				f"Parallel tokenizing {num_sequences} sequences across {self.state.world_size} ranks "
				f"(~{len(my_indices)} per rank)..."
			)

		tokenize_start = time.perf_counter()

		# Each rank tokenizes its subset
		if my_texts:
			my_batch_tokenized = self.state.tokenizer(
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
		logging.debug(f"Rank {self.state.rank}: Local tokenization of {len(my_texts)} sequences in {local_tokenize_time:.2f}s")

		# DEBUG: Print tokenized prompts
		if os.environ.get("BATCHGEN_DEBUG_TOKENIZE", "0") == "1" and self.state.rank == 0 and my_tokenized:
			print(f"\n[TOKENIZE DEBUG] === First 3 tokenized prompts ===")
			for i in range(min(3, len(my_tokenized))):
				item = my_tokenized[i]
				token_ids = item["input_ids"]
				print(f"\n[TOKENIZE DEBUG] Sequence {item['global_idx']} (length={item['length']})")
				print(f"[TOKENIZE DEBUG] First 50 tokens: {token_ids[:50]}")
				print(f"[TOKENIZE DEBUG] Last 50 tokens: {token_ids[-50:]}")
				try:
					decoded_start = self.state.tokenizer.decode(token_ids[:100])
					decoded_end = self.state.tokenizer.decode(token_ids[-100:])
					print(f"[TOKENIZE DEBUG] Start of prompt (decoded): {repr(decoded_start[:300])}")
					print(f"[TOKENIZE DEBUG] End of prompt (decoded): {repr(decoded_end[-300:])}")
				except Exception as e:
					print(f"[TOKENIZE DEBUG] Decode error: {e}")
				special_token_ids = [199998, 199999, 200000, 200001, 200002, 200003, 200004, 200005, 200006, 200007, 200008, 200012]
				found_special = [tid for tid in token_ids if tid in special_token_ids]
				if found_special:
					print(f"[TOKENIZE DEBUG] Special tokens found: {found_special}")

		# Phase 1.5: Gather all tokenized results to all ranks
		# This keeps NCCL alive and shares results efficiently
		gather_start = time.perf_counter()
		all_tokenized_lists = [None] * self.state.world_size
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
		if self.state.rank == 0:
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
		self.state.rejected_sequences = []
		uuids_to_remove = []
		for seq in self.state.global_batch:
			pl = tokenized_by_idx[seq.global_idx]["length"]
			if pl >= self.state.model_context_length:
				self.state.rejected_sequences.append((seq.global_idx, pl))
				uuids_to_remove.append(seq.uuid)
				# Free tokenized data for rejected sequence
				del tokenized_by_idx[seq.global_idx]

		for uuid in uuids_to_remove:
			self.state.global_batch.remove_sequence(uuid)

		if self.state.rejected_sequences:
			logging.info(
				f"Rank {self.state.rank}: Rejected {len(self.state.rejected_sequences)}/"
				f"{len(self.state.rejected_sequences) + len(self.state.global_batch)} "
				f"sequences exceeding context length {self.state.model_context_length}"
			)

		# Recalculate max_prompt_length after rejection (remaining sequences only)
		num_sequences = len(self.state.global_batch)
		if num_sequences > 0:
			remaining_lengths = [tokenized_by_idx[seq.global_idx]["length"] for seq in self.state.global_batch]
			max_prompt_length = max(remaining_lengths)
		else:
			max_prompt_length = 0

		# Update self.state.max_input_length to the actual longest prompt
		# This is used for attention mask shape: [bsz, max_prompt_length + max_decoding_length]
		self.state.max_input_length = max_prompt_length
		if num_sequences > 0:
			logging.info(
				f"Rank {self.state.rank}: Dynamic max_prompt_length set to {max_prompt_length} "
				f"(prompt lengths: min={min(remaining_lengths)}, max={max(remaining_lengths)}, "
				f"count={num_sequences})"
			)

		# Phase 3: Create per-sequence tensor views from pre-allocated buffer pool.
		# Pre-allocating 2 large contiguous buffers eliminates allocator contention
		# when 16 ranks run Phase 3 simultaneously (was 192K allocations → now 32).
		# Skip if all sequences were rejected in Phase 2.5.
		if num_sequences == 0:
			logging.info(f"Rank {self.state.rank}: All sequences rejected, skipping Phase 3 buffer allocation")
			return

		phase3_start = time.perf_counter()
		num_seqs = len(self.state.global_batch)

		self.state.buffer_pool = QueryBookBufferPool(
			num_sequences=num_seqs,
			model_context_length=self.state.model_context_length,
			max_decoding_length=self.state.max_decoding_length,
			pad_token_id=self.state.pad_token_id,
		)
		t_alloc = time.perf_counter() - phase3_start
		logging.info(
			f"Rank {self.state.rank}: Phase 3 buffer pool allocated in {t_alloc:.2f}s "
			f"(input_ids: [{num_seqs}, {self.state.model_context_length}], "
			f"decoded_tokens: [{num_seqs}, {self.state.max_decoding_length}])"
		)

		for seq_i, seq in enumerate(self.state.global_batch):
			item = tokenized_by_idx[seq.global_idx]
			input_ids_list = item["input_ids"]
			actual_prompt_len = item["length"]

			if len(input_ids_list) != actual_prompt_len:
				logging.error(
					f"Rank {self.state.rank}: Token length mismatch for seq {seq.global_idx}: "
					f"list_len={len(input_ids_list)}, stored_len={actual_prompt_len}"
				)
				actual_prompt_len = len(input_ids_list)

			seq_extended_size = min(
				actual_prompt_len + self.state.max_decoding_length,
				self.state.model_context_length
			)

			slot = self.state.buffer_pool.allocate_slot()
			seq._buffer_slot = slot

			input_ids_view = self.state.buffer_pool.get_input_ids_view(slot, seq_extended_size)
			input_ids_view[0, :actual_prompt_len] = torch.tensor(input_ids_list, dtype=torch.long)
			seq.input_ids = input_ids_view
			seq.decoded_tokens = self.state.buffer_pool.get_decoded_tokens_view(slot)

			# Free the tokenized data for this sequence immediately
			del tokenized_by_idx[seq.global_idx]

			seq.prompt_length = actual_prompt_len
			seq.original_prompt_length = actual_prompt_len  # Must match prompt_length at tokenization time
			seq.current_context_length = actual_prompt_len
			seq.kv_token_budget = seq_extended_size

			if (seq_i + 1) % 3000 == 0:
				elapsed = time.perf_counter() - phase3_start
				logging.info(
					f"Rank {self.state.rank}: Phase 3 progress: {seq_i+1}/{num_seqs} sequences "
					f"({elapsed:.1f}s elapsed)"
				)

		phase3_total = time.perf_counter() - phase3_start
		logging.info(
			f"Rank {self.state.rank}: Phase 3 complete: {num_seqs} sequences in {phase3_total:.2f}s "
			f"(buffer alloc: {t_alloc:.2f}s, fill: {phase3_total-t_alloc:.2f}s)"
		)

		logging.info(f"Rank {self.state.rank}: Tokenized {len(self.state.global_batch)} sequences")

	def assign_to_ranks(self) -> None:
		"""
		Assign sequences to ranks balancing predicted attention tile workload.
		All ranks execute this identically to maintain consistent assignment.

		Uses greedy bin-packing: sort sequences by predicted tiles (descending),
		then assign each to the rank with fewest total tiles. This balances
		attention compute across ranks, reducing synchronization wait time.
		"""
		if self.state.global_batch is None:
			raise RuntimeError("Global batch not initialized")

		# Sort sequences by predicted total context (descending) for better bin-packing
		# Larger sequences first ensures better balance
		sequences = list(self.state.global_batch)
		sequences.sort(
			key=lambda s: s.prompt_length + s.max_decode_length,
			reverse=True
		)

		# Track total tiles per rank (attention tile = 128 tokens)
		TILE_SIZE = 128
		rank_tiles = [0] * self.state.world_size

		for seq in sequences:
			# Predict total context length at decode completion
			predicted_context = seq.prompt_length + seq.max_decode_length
			predicted_tiles = (predicted_context + TILE_SIZE - 1) // TILE_SIZE  # ceil_div

			# Assign to rank with fewest tiles (greedy)
			target_rank = rank_tiles.index(min(rank_tiles))
			self.state.global_batch.assign_rank(seq.uuid, target_rank)
			rank_tiles[target_rank] += predicted_tiles

		# Log balance quality
		my_seqs = self.state.global_batch.get_sequences_for_rank(self.state.rank)
		if self.state.rank == 0:
			imbalance = (max(rank_tiles) - min(rank_tiles)) / max(rank_tiles) * 100 if max(rank_tiles) > 0 else 0
			logging.info(
				f"Workload distribution (tiles per rank): {rank_tiles}, "
				f"imbalance: {imbalance:.1f}%"
			)
		logging.info(
			f"Rank {self.state.rank}: Assigned {len(my_seqs)} sequences, "
			f"tiles={rank_tiles[self.state.rank]}"
		)

	def build_query_book(self) -> None:
		"""
		Build query_book from global_batch for sequences assigned to this rank.
		Maps local indices (0, 1, 2, ...) to sequence data for backward compatibility.
		"""
		my_uuids = sorted(
			self.state.global_batch.get_sequences_for_rank(self.state.rank),
			key=lambda uuid: self.state.global_batch.get_sequence(uuid).global_idx
		)

		self.state.query_book = {}
		self.state.local_to_uuid_map = {}
		self.state.uuid_to_local_map = {}
		self.state.free_local_indices = set()  # Reset free list
		self.state.next_local_idx = len(my_uuids)  # Next available index after initial assignment

		for local_idx, uuid in enumerate(my_uuids):
			seq = self.state.global_batch.get_sequence(uuid)

			self.state.query_book[local_idx] = query(
				text=seq.text,
				encoded={
					"input_ids": seq.input_ids,
				},
				decoded_tokens=seq.decoded_tokens,
				kv_token_budget=seq.kv_token_budget,
			)

			self.state.local_to_uuid_map[local_idx] = uuid
			self.state.uuid_to_local_map[uuid] = local_idx

		# Validation: Check that we have all sequences assigned to this rank
		expected_count = sum(
			1 for seq in self.state.global_batch if seq.assigned_rank == self.state.rank
		)

		if len(my_uuids) != expected_count:
			logging.error(
				f"Rank {self.state.rank}: CRITICAL MISMATCH - expected {expected_count} sequences "
				f"but got {len(my_uuids)} from get_sequences_for_rank!"
			)

		logging.info(
			f"Rank {self.state.rank}: Built local query_book with {len(self.state.query_book)} entries "
			f"(global_batch has {len(self.state.global_batch)} sequences)"
		)
