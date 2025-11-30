# """
# 	Data structure for storing and updating query information.
# """
# from enum import IntEnum
# class SequenceStatus(IntEnum):
# 	QUEUEING = 0
# 	IN_PREFILL = 1
# 	PREFILLED = 2
# 	IN_DECODE = 3
# 	ON_HOLD = 4
# 	COMPLETED = 5

# class SequenceEntry:
# 	__slots__ = ('uuid', 'prompt_length', 'max_decode_length', 
# 			  	'status', 'decoded_length', 'current_context_length',
# 				'input_ids')

# 	VALID_TRANSITIONS = {
# 			SequenceStatus.QUEUEING: {SequenceStatus.IN_PREFILL},
# 			SequenceStatus.IN_PREFILL: {SequenceStatus.PREFILLED}, # Currently the prefill would not be interrupted.
# 			SequenceStatus.PREFILLED: {SequenceStatus.IN_DECODE, SequenceStatus.ON_HOLD},
# 			SequenceStatus.IN_DECODE: {SequenceStatus.ON_HOLD, SequenceStatus.COMPLETED},
# 			SequenceStatus.ON_HOLD: {SequenceStatus.IN_DECODE},
# 			SequenceStatus.COMPLETED: set(),  # No transitions allowed from COMPLETED
# 	}	

# 	def __init__(self, uuid:str, prompt_length:int, max_decode_length:int):
# 		self.uuid = uuid
# 		self.prompt_length = prompt_length
# 		self.max_decode_length = max_decode_length
# 		self.status = SequenceStatus.QUEUEING
# 		self.decoded_length = 0
# 		self.current_context_length = prompt_length
# 		self.input_ids = None  # To be set when the query is processed
	
# 	def status_transition(self, new_status:SequenceStatus):
# 		if new_status in self.VALID_TRANSITIONS[self.status]:
# 			self.status = new_status
# 		else:
# 			raise ValueError(f"Invalid status transition from {self.status} to {new_status}")
	

# class SequenceBatch:
# 	__slots__ = ('sequences')
# 	def __init__(self):
# 		self.sequences = {} # uuid -> SequenceEntry
# 	def add_sequence(self, sequence:SequenceEntry):
# 		self.sequences[sequence.uuid] = sequence
# 	def get_sequence(self, uuid:str) -> SequenceEntry:
# 		return self.sequences.get(uuid, None)
# 	def update_status(self, uuid:str, new_status:SequenceStatus):
# 		if uuid in self.sequences:
# 			self.sequences[uuid].status = new_status
# 		else:
# 			raise KeyError(f"Sequence with UUID {uuid} not found.")
# 	def create_view(self, uuids: list[str]) -> 'SequenceBatch':
# 		"""
# 		Create a view batch containing only specified UUIDs.
# 		Returns references to the same sequence objects.
		
# 		Args:
# 			uuids: List of UUIDs to include in the view
		
# 		Returns:
# 			New SequenceBatch with references to specified sequences
# 		"""
# 		view = SequenceBatch()
# 		for uuid in uuids:
# 			if uuid in self.sequences:
# 				view.sequences[uuid] = self.sequences[uuid]  # Reference!
# 		return view


"""
    Data structure for storing and updating query information.
"""
from enum import IntEnum
from typing import Dict, List, Optional, Set
import torch


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
        'kv_token_budget', 'assigned_rank', 'text'
    )

    VALID_TRANSITIONS = {
        SequenceStatus.QUEUEING: {SequenceStatus.IN_PREFILL},
        SequenceStatus.IN_PREFILL: {SequenceStatus.PREFILLED},
        SequenceStatus.PREFILLED: {SequenceStatus.IN_DECODE, SequenceStatus.ON_HOLD},
        SequenceStatus.IN_DECODE: {SequenceStatus.ON_HOLD, SequenceStatus.COMPLETED},
        SequenceStatus.ON_HOLD: {SequenceStatus.IN_DECODE},
        SequenceStatus.COMPLETED: set(),
    }

    def __init__(
        self,
        uuid: str,
        global_idx: int,
        prompt_length: int,
        max_decode_length: int,
        text: Optional[str] = None,
    ):
        self.uuid = uuid
        self.global_idx = global_idx  # Position in global batch (0, 1, 2, ...)
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
        self.text = text  # Original prompt text (for debugging)

    def status_transition(self, new_status: SequenceStatus) -> None:
        if new_status in self.VALID_TRANSITIONS[self.status]:
            self.status = new_status
        else:
            raise ValueError(
                f"Invalid status transition from {self.status.name} to {new_status.name} "
                f"for sequence {self.uuid}"
            )

    def is_finished(self) -> bool:
        """Check if sequence has completed generation."""
        return self.status == SequenceStatus.COMPLETED

    def remaining_decode_tokens(self) -> int:
        """Return number of tokens left to decode."""
        return self.max_decode_length - self.decoded_length


class SequenceBatch:
    __slots__ = ('sequences', '_status_index', '_rank_index')

    def __init__(self):
        self.sequences: Dict[str, SequenceEntry] = {}
        # Index structures for efficient lookups
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
        seq.status_transition(new_status)  # This validates the transition
        
        # Update index
        self._status_index[old_status].discard(uuid)
        self._status_index[new_status].add(uuid)

    def assign_rank(self, uuid: str, rank: int) -> None:
        """Assign a sequence to a specific rank."""
        if uuid not in self.sequences:
            raise KeyError(f"Sequence with UUID {uuid} not found.")
        
        seq = self.sequences[uuid]
        old_rank = seq.assigned_rank
        
        # Update rank index
        if old_rank is not None and old_rank in self._rank_index:
            self._rank_index[old_rank].discard(uuid)
        
        if rank not in self._rank_index:
            self._rank_index[rank] = set()
        self._rank_index[rank].add(uuid)
        
        seq.assigned_rank = rank

    def get_sequences_by_status(self, status: SequenceStatus) -> List[str]:
        """Get list of UUIDs with given status."""
        return list(self._status_index[status])

    def get_sequences_for_rank(self, rank: int) -> List[str]:
        """Get list of UUIDs assigned to a specific rank."""
        return list(self._rank_index.get(rank, set()))

    def get_sequences_for_rank_with_status(
        self, rank: int, status: SequenceStatus
    ) -> List[str]:
        """Get UUIDs assigned to rank with specific status."""
        rank_seqs = self._rank_index.get(rank, set())
        status_seqs = self._status_index[status]
        return list(rank_seqs & status_seqs)

    def create_view(self, uuids: List[str]) -> 'SequenceBatch':
        """
        Create a view batch containing only specified UUIDs.
        Returns references to the same sequence objects.
        """
        view = SequenceBatch()
        for uuid in uuids:
            if uuid in self.sequences:
                seq = self.sequences[uuid]
                view.sequences[uuid] = seq  # Reference, not copy
                view._status_index[seq.status].add(uuid)
                if seq.assigned_rank is not None:
                    if seq.assigned_rank not in view._rank_index:
                        view._rank_index[seq.assigned_rank] = set()
                    view._rank_index[seq.assigned_rank].add(uuid)
        return view

    # Convenience methods for loop control
    def has_queueing(self) -> bool:
        return len(self._status_index[SequenceStatus.QUEUEING]) > 0

    def has_prefilled(self) -> bool:
        return len(self._status_index[SequenceStatus.PREFILLED]) > 0

    def has_in_decode(self) -> bool:
        return len(self._status_index[SequenceStatus.IN_DECODE]) > 0

    def has_on_hold(self) -> bool:
        return len(self._status_index[SequenceStatus.ON_HOLD]) > 0

    def all_completed(self) -> bool:
        return len(self._status_index[SequenceStatus.COMPLETED]) == len(self.sequences)

    def count_by_status(self, status: SequenceStatus) -> int:
        return len(self._status_index[status])

    def __len__(self) -> int:
        return len(self.sequences)

    def __iter__(self):
        return iter(self.sequences.values())