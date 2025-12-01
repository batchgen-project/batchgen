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
        'kv_token_budget', 'assigned_rank', 'text', 'eos_reached'
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
        self.eos_reached: bool = False

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
        """Return number of pages currently used by this sequence."""
        return math.ceil(self.current_context_length / self.PAGE_SIZE)

    def is_finished(self) -> bool:
        return self.status == SequenceStatus.COMPLETED

    def remaining_decode_tokens(self) -> int:
        return self.max_decode_length - self.decoded_length

    def should_check_completion(self) -> bool:
        """Check if we're at a page boundary (every PAGE_SIZE tokens in decoding)."""
        return self.decoded_length > 0 and self.decoded_length % self.PAGE_SIZE == 0

    def check_completion(self, eos_token_id: int) -> bool:
        """
        Check if sequence should be marked as completed.
        Returns True if sequence is complete (EOS reached or max length).
        """
        # Check max decoding length
        if self.decoded_length >= self.max_decode_length:
            return True
        
        # Check EOS token
        if self.eos_reached:
            return True
            
        return False


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

    def has_in_decode(self) -> bool:
        return len(self._status_index[SequenceStatus.IN_DECODE]) > 0

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

    def get_current_pages_used_for_sequences(self, uuids: List[str]) -> int:
        """Compute total pages currently used by given sequences."""
        total = 0
        for uuid in uuids:
            seq = self.sequences.get(uuid)
            if seq:
                total += seq.get_current_pages_used()
        return total

    def __len__(self) -> int:
        return len(self.sequences)

    def __iter__(self):
        return iter(self.sequences.values())