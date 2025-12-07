"""
    Data structure for storing and updating query information.
    
    Supports two-page buffer design for efficient GPU KV cache management:
    - Instead of allocating full (prompt + max_decode) pages upfront,
      we allocate current_context_length + 2*PAGE_SIZE tokens worth of pages.
    - New KV tokens are streamed to host after each attention layer.
    - Sequences can be put ON_HOLD when GPU memory is insufficient.
"""
from enum import IntEnum
from typing import Dict, List, Optional, Set
import torch
import math


class SequenceStatus(IntEnum):
    QUEUEING = 0      # Waiting to be prefilled
    IN_PREFILL = 1    # Currently being prefilled
    PREFILLED = 2     # Prefilled, KV in host, waiting for GPU
    IN_DECODE = 3     # Currently decoding with KV in GPU
    ON_HOLD = 4       # Was decoding, paused due to GPU memory pressure, KV in host
    COMPLETED = 5     # Finished generation


class SequenceEntry:
    __slots__ = (
        'uuid', 'global_idx', 'prompt_length', 'max_decode_length',
        'status', 'decoded_length', 'current_context_length',
        'input_ids', 'attention_mask', 'decoded_tokens',
        'kv_token_budget', 'assigned_rank', 'text', 'eos_reached',
        # Two-page buffer tracking
        'gpu_pages_allocated',
    )

    VALID_TRANSITIONS = {
        SequenceStatus.QUEUEING: {SequenceStatus.IN_PREFILL},
        SequenceStatus.IN_PREFILL: {SequenceStatus.PREFILLED},
        SequenceStatus.PREFILLED: {SequenceStatus.IN_DECODE},
        SequenceStatus.IN_DECODE: {SequenceStatus.ON_HOLD, SequenceStatus.COMPLETED},
        SequenceStatus.ON_HOLD: {SequenceStatus.IN_DECODE, SequenceStatus.COMPLETED},
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
        
        # Two-page buffer: tracks current GPU page allocation
        # This is separate from host pages (which always hold full kv_token_budget)
        self.gpu_pages_allocated: int = 0

    def status_transition(self, new_status: SequenceStatus) -> None:
        if new_status in self.VALID_TRANSITIONS[self.status]:
            self.status = new_status
        else:
            raise ValueError(
                f"Invalid status transition from {self.status.name} to {new_status.name} "
                f"for sequence {self.uuid}"
            )

    # ============ Full KV Budget Methods (Host KV) ============

    def get_pages_required(self) -> int:
        """
        Return total pages required for this sequence's full KV cache.
        Used for HOST KV allocation (which holds complete KV history).
        """
        return math.ceil(self.kv_token_budget / self.PAGE_SIZE)

    def get_current_pages_used(self) -> int:
        """Return number of pages currently used by this sequence's context."""
        return math.ceil(self.current_context_length / self.PAGE_SIZE)

    # ============ Two-Page Buffer Methods (GPU KV) ============

    def get_gpu_pages_for_two_page_buffer(self) -> int:
        """
        Get GPU pages needed for two-page buffer design.
        
        Allocates enough pages for: current_context_length + 2*PAGE_SIZE tokens.
        This ensures we have a 2-page (128 token) runway before needing to
        either extend allocation or put the sequence ON_HOLD.
        
        Returns:
            Number of pages to allocate on GPU
        """
        buffer_tokens = self.current_context_length + 2 * self.PAGE_SIZE
        # Cap at full budget (don't over-allocate)
        buffer_tokens = min(buffer_tokens, self.kv_token_budget)
        return math.ceil(buffer_tokens / self.PAGE_SIZE)

    def get_additional_gpu_pages_needed(self) -> int:
        """
        Get additional GPU pages needed to maintain two-page buffer.
        
        Called at page boundaries to check if we need to extend GPU allocation.
        
        Returns:
            Number of additional pages needed (0 if current allocation is sufficient)
        """
        required = self.get_gpu_pages_for_two_page_buffer()
        return max(0, required - self.gpu_pages_allocated)

    def get_gpu_page_headroom(self) -> int:
        """
        Get number of tokens we can decode before hitting GPU page limit.
        
        Returns:
            Number of tokens until we exhaust current GPU page allocation
        """
        allocated_tokens = self.gpu_pages_allocated * self.PAGE_SIZE
        return max(0, allocated_tokens - self.current_context_length)

    def tokens_until_next_page_boundary(self) -> int:
        """
        Get tokens until next page boundary in current context.
        
        Returns:
            Number of tokens until current_context_length crosses a page boundary
        """
        tokens_in_current_page = self.current_context_length % self.PAGE_SIZE
        if tokens_in_current_page == 0:
            return self.PAGE_SIZE
        return self.PAGE_SIZE - tokens_in_current_page

    def needs_gpu_page_extension(self) -> bool:
        """
        Check if sequence needs GPU page extension to maintain two-page buffer.
        
        Returns:
            True if additional GPU pages are needed
        """
        return self.get_additional_gpu_pages_needed() > 0

    def reset_gpu_allocation(self) -> None:
        """
        Reset GPU page allocation tracking.
        Called when sequence is put ON_HOLD (GPU pages released, host KV retained).
        """
        self.gpu_pages_allocated = 0

    # ============ Completion and Status Methods ============

    def is_finished(self) -> bool:
        return self.status == SequenceStatus.COMPLETED

    def is_active_in_decode(self) -> bool:
        """Check if sequence is actively decoding (not on hold or completed)."""
        return self.status == SequenceStatus.IN_DECODE

    def is_resumable(self) -> bool:
        """Check if sequence can be resumed from ON_HOLD or loaded from PREFILLED."""
        return self.status in (SequenceStatus.PREFILLED, SequenceStatus.ON_HOLD)

    def remaining_decode_tokens(self) -> int:
        return self.max_decode_length - self.decoded_length

    def should_check_completion(self) -> bool:
        """Check if we're at a page boundary (every PAGE_SIZE tokens in decoding)."""
        return self.decoded_length > 0 and self.decoded_length % self.PAGE_SIZE == 0

    def check_completion(self, eos_token_id: int) -> bool:
        """
        Check if sequence should be marked as completed.
        
        Args:
            eos_token_id: The EOS token ID to check against
            
        Returns:
            True if sequence is complete (EOS reached or max length)
        """
        if self.decoded_length >= self.max_decode_length:
            return True
        
        if self.eos_reached:
            return True
            
        return False

    def __repr__(self) -> str:
        return (
            f"SequenceEntry(uuid={self.uuid}, global_idx={self.global_idx}, "
            f"status={self.status.name}, decoded={self.decoded_length}/{self.max_decode_length}, "
            f"ctx_len={self.current_context_length}, gpu_pages={self.gpu_pages_allocated})"
        )


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

    def get_resumable_sequences(self) -> List[str]:
        """
        Get sequences that can be loaded/resumed into GPU decode.
        Includes both PREFILLED (never started decode) and ON_HOLD (paused).
        
        Returns:
            List of UUIDs for resumable sequences
        """
        prefilled = self._status_index[SequenceStatus.PREFILLED]
        on_hold = self._status_index[SequenceStatus.ON_HOLD]
        return list(prefilled | on_hold)

    def get_resumable_sequences_for_rank(self, rank: int) -> List[str]:
        """
        Get resumable sequences assigned to a specific rank.
        
        Args:
            rank: The rank to filter by
            
        Returns:
            List of UUIDs for resumable sequences on this rank
        """
        rank_seqs = self._rank_index.get(rank, set())
        prefilled = self._status_index[SequenceStatus.PREFILLED]
        on_hold = self._status_index[SequenceStatus.ON_HOLD]
        resumable = prefilled | on_hold
        return list(rank_seqs & resumable)

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

    # ============ Convenience Status Check Methods ============

    def has_queueing(self) -> bool:
        return len(self._status_index[SequenceStatus.QUEUEING]) > 0

    def has_prefilled(self) -> bool:
        return len(self._status_index[SequenceStatus.PREFILLED]) > 0

    def has_in_decode(self) -> bool:
        return len(self._status_index[SequenceStatus.IN_DECODE]) > 0

    def has_on_hold(self) -> bool:
        """Check if there are sequences paused due to GPU memory pressure."""
        return len(self._status_index[SequenceStatus.ON_HOLD]) > 0

    def has_resumable(self) -> bool:
        """Check if there are sequences that can be loaded/resumed into GPU."""
        return self.has_prefilled() or self.has_on_hold()

    def all_completed(self) -> bool:
        return len(self._status_index[SequenceStatus.COMPLETED]) == len(self.sequences)

    def count_by_status(self, status: SequenceStatus) -> int:
        return len(self._status_index[status])

    # ============ Page Calculation Methods ============

    def get_total_pages_for_sequences(self, uuids: List[str]) -> int:
        """Compute total pages required for given sequences (full KV budget)."""
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

    def get_two_page_buffer_pages_for_sequences(self, uuids: List[str]) -> int:
        """
        Compute total GPU pages needed for two-page buffer design.
        
        Args:
            uuids: List of sequence UUIDs
            
        Returns:
            Total pages needed for all sequences with two-page buffer
        """
        total = 0
        for uuid in uuids:
            seq = self.sequences.get(uuid)
            if seq:
                total += seq.get_gpu_pages_for_two_page_buffer()
        return total

    def get_total_gpu_pages_allocated(self, uuids: List[str]) -> int:
        """
        Get total GPU pages currently allocated for given sequences.
        
        Args:
            uuids: List of sequence UUIDs
            
        Returns:
            Total GPU pages allocated
        """
        total = 0
        for uuid in uuids:
            seq = self.sequences.get(uuid)
            if seq:
                total += seq.gpu_pages_allocated
        return total

    def get_sequences_needing_gpu_extension(self, uuids: List[str]) -> List[str]:
        """
        Get sequences that need GPU page extension to maintain two-page buffer.
        
        Args:
            uuids: List of sequence UUIDs to check
            
        Returns:
            List of UUIDs that need extension
        """
        need_extension = []
        for uuid in uuids:
            seq = self.sequences.get(uuid)
            if seq and seq.needs_gpu_page_extension():
                need_extension.append(uuid)
        return need_extension

    # ============ Batch Operations ============

    def reset_gpu_allocations_for_sequences(self, uuids: List[str]) -> None:
        """
        Reset GPU page allocation tracking for sequences being put ON_HOLD.
        
        Args:
            uuids: List of sequence UUIDs
        """
        for uuid in uuids:
            seq = self.sequences.get(uuid)
            if seq:
                seq.reset_gpu_allocation()

    def __len__(self) -> int:
        return len(self.sequences)

    def __iter__(self):
        return iter(self.sequences.values())

    def __repr__(self) -> str:
        status_counts = {s.name: self.count_by_status(s) for s in SequenceStatus}
        return f"SequenceBatch(total={len(self)}, {status_counts})"