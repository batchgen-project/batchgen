"""
	Data structure for storing and updating query information.
"""
from enum import IntEnum
from typing import Dict, List, Optional, Set
import torch
import math


class SequenceStatus(IntEnum):
	QUEUEING = 0
	IN_PREFILL = 1
	PREFILLED = 2
	IN_DECODE = 3
	ON_HOLD = 4
	COMPLETED = 5


class SequenceEntry:
	__slots__ = (
		'uuid', 'global_idx', 'prompt_length', 'max_decode_length',
		'status', 'decoded_length', 'current_context_length',
		'input_ids', 'attention_mask', 'decoded_tokens',
		'kv_token_budget', 'assigned_rank', 'text',
		'eos_reached', 'last_token_id',  # NEW: for continuous batching
	)

	VALID_TRANSITIONS = {
		SequenceStatus.QUEUEING: {SequenceStatus.IN_PREFILL},
		SequenceStatus.IN_PREFILL: {SequenceStatus.PREFILLED},
		SequenceStatus.PREFILLED: {SequenceStatus.IN_DECODE, SequenceStatus.ON_HOLD},
		SequenceStatus.IN_DECODE: {SequenceStatus.ON_HOLD, SequenceStatus.COMPLETED},
		SequenceStatus.ON_HOLD: {SequenceStatus.IN_DECODE},
		SequenceStatus.COMPLETED: set(),
	}

	# Page size for KV cache (tokens per page)
	PAGE_SIZE = 64

	def __init__(
		self,
		uuid: str,
		global_idx: int,
		prompt_length: int,
		max_decode_length: int,
		text: Optional[str] = None,
	):
		self.uuid = uuid
		self.global_idx = global_idx
		self.prompt_length = prompt_length
		self.max_decode_length = max_decode_length
		self.status = SequenceStatus.QUEUEING
		self.decoded_length = 0
		self.current_context_length = prompt_length
		self.input_ids: Optional[torch.Tensor] = None
		self.attention_mask: Optional[torch.Tensor] = None
		self.decoded_tokens: Optional[torch.Tensor] = None
		self.kv_token_budget: int = prompt_length + max_decode_length
		self.assigned_rank: Optional[int] = None
		self.text = text
		# NEW: for continuous batching
		self.eos_reached = False
		self.last_token_id: Optional[int] = None

	def status_transition(self, new_status: SequenceStatus) -> None:
		if new_status in self.VALID_TRANSITIONS[self.status]:
			self.status = new_status
		else:
			raise ValueError(
				f"Invalid status transition from {self.status.name} to {new_status.name} "
				f"for sequence {self.uuid}"
			)

	def get_pages_required(self) -> int:
		"""Return number of pages required for this sequence's KV cache."""
		return math.ceil(self.kv_token_budget / self.PAGE_SIZE)

	def get_current_pages_used(self) -> int:
		"""Return number of pages currently used based on context length."""
		return math.ceil(self.current_context_length / self.PAGE_SIZE)

	def is_finished(self) -> bool:
		return self.status == SequenceStatus.COMPLETED

	def remaining_decode_tokens(self) -> int:
		return self.max_decode_length - self.decoded_length

	# NEW: Continuous batching helpers
	def should_continue_decoding(self, eos_token_id: int) -> bool:
		"""Check if sequence should continue decoding."""
		if self.eos_reached:
			return False
		if self.decoded_length >= self.max_decode_length:
			return False
		return True

	def check_and_mark_eos(self, token_id: int, eos_token_id: int) -> bool:
		"""
		Check if token is EOS and mark accordingly.
		Returns True if EOS was detected.
		"""
		self.last_token_id = token_id
		if token_id == eos_token_id:
			self.eos_reached = True
			return True
		return False

	def increment_decoded_length(self, count: int = 1) -> None:
		"""Increment decoded length and update context length."""
		self.decoded_length += count
		self.current_context_length = self.prompt_length + self.decoded_length


class SequenceBatch:
	__slots__ = ('sequences', '_status_index', '_rank_index')

	def __init__(self):
		self.sequences: Dict[str, SequenceEntry] = {}
		self._status_index: Dict[SequenceStatus, Set[str]] = {
			status: set() for status in SequenceStatus
		}
		self._rank_index: Dict[int, Set[str]] = {}

	def add_sequence(self, sequence: SequenceEntry) -> None:
		self.sequences[sequence.uuid] = sequence
		self._status_index[sequence.status].add(sequence.uuid)
		if sequence.assigned_rank is not None:
			if sequence.assigned_rank not in self._rank_index:
				self._rank_index[sequence.assigned_rank] = set()
			self._rank_index[sequence.assigned_rank].add(sequence.uuid)

	def get_sequence(self, uuid: str) -> Optional[SequenceEntry]:
		return self.sequences.get(uuid, None)

	def update_status(self, uuid: str, new_status: SequenceStatus) -> None:
		if uuid not in self.sequences:
			raise KeyError(f"Sequence with UUID {uuid} not found.")
		
		seq = self.sequences[uuid]
		old_status = seq.status
		seq.status_transition(new_status)  # Validates transition
		
		# Update index
		self._status_index[old_status].discard(uuid)
		self._status_index[new_status].add(uuid)

	def assign_rank(self, uuid: str, rank: int) -> None:
		if uuid not in self.sequences:
			raise KeyError(f"Sequence with UUID {uuid} not found.")
		
		seq = self.sequences[uuid]
		old_rank = seq.assigned_rank
		
		if old_rank is not None and old_rank in self._rank_index:
			self._rank_index[old_rank].discard(uuid)
		
		if rank not in self._rank_index:
			self._rank_index[rank] = set()
		self._rank_index[rank].add(uuid)
		
		seq.assigned_rank = rank

	def get_sequences_by_status(self, status: SequenceStatus) -> List[str]:
		return list(self._status_index[status])

	def get_sequences_for_rank(self, rank: int) -> List[str]:
		return list(self._rank_index.get(rank, set()))

	def get_sequences_for_rank_with_status(
		self, rank: int, status: SequenceStatus
	) -> List[str]:
		rank_seqs = self._rank_index.get(rank, set())
		status_seqs = self._status_index[status]
		return list(rank_seqs & status_seqs)

	def create_view(self, uuids: List[str]) -> 'SequenceBatch':
		view = SequenceBatch()
		for uuid in uuids:
			if uuid in self.sequences:
				seq = self.sequences[uuid]
				view.sequences[uuid] = seq
				view._status_index[seq.status].add(uuid)
				if seq.assigned_rank is not None:
					if seq.assigned_rank not in view._rank_index:
						view._rank_index[seq.assigned_rank] = set()
					view._rank_index[seq.assigned_rank].add(uuid)
		return view

	# Convenience methods
	def has_queueing(self) -> bool:
		return len(self._status_index[SequenceStatus.QUEUEING]) > 0

	def has_prefilled(self) -> bool:
		return len(self._status_index[SequenceStatus.PREFILLED]) > 0

	def all_completed(self) -> bool:
		return len(self._status_index[SequenceStatus.COMPLETED]) == len(self.sequences)

	def count_by_status(self, status: SequenceStatus) -> int:
		return len(self._status_index[status])

	def get_total_pages_for_sequences(self, uuids: List[str]) -> int:
		"""Compute total pages required for given sequences."""
		total = 0
		for uuid in uuids:
			seq = self.sequences.get(uuid)
			if seq:
				total += seq.get_pages_required()
		return total

	# NEW: Continuous batching helpers
	def has_in_decode(self) -> bool:
		"""Check if there are sequences currently being decoded."""
		return len(self._status_index[SequenceStatus.IN_DECODE]) > 0

	def has_loadable_sequences(self) -> bool:
		"""Check if there are sequences that can be loaded into decode batch."""
		return (len(self._status_index[SequenceStatus.PREFILLED]) > 0 or
				len(self._status_index[SequenceStatus.ON_HOLD]) > 0)

	def get_in_decode_sequences(self) -> List[str]:
		"""Get all sequences currently in decode status."""
		return list(self._status_index[SequenceStatus.IN_DECODE])

	def get_loadable_sequences(self) -> List[str]:
		"""
		Get sequences that can be loaded into the decode batch.
		Prioritizes ON_HOLD (previously paused) over PREFILLED (new).
		Returns sorted by global_idx for deterministic ordering.
		"""
		on_hold = list(self._status_index[SequenceStatus.ON_HOLD])
		prefilled = list(self._status_index[SequenceStatus.PREFILLED])
		
		# Sort each by global_idx for deterministic ordering
		on_hold.sort(key=lambda uuid: self.sequences[uuid].global_idx)
		prefilled.sort(key=lambda uuid: self.sequences[uuid].global_idx)
		
		# Prioritize ON_HOLD sequences
		return on_hold + prefilled

	def get_sequences_to_complete(self, eos_token_id: int) -> List[str]:
		"""
		Get IN_DECODE sequences that should be marked as completed.
		A sequence should complete if:
		- EOS token was reached, OR
		- decoded_length >= max_decode_length
		"""
		in_decode = self._status_index[SequenceStatus.IN_DECODE]
		to_complete = []
		for uuid in in_decode:
			seq = self.sequences[uuid]
			if not seq.should_continue_decoding(eos_token_id):
				to_complete.append(uuid)
		return to_complete

	def __len__(self) -> int:
		return len(self.sequences)

	def __iter__(self):
		return iter(self.sequences.values())