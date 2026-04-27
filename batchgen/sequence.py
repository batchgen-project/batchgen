"""
    Data structure for storing and updating query information.

    Supports configurable page buffer design for efficient GPU KV cache management:
    - Initial reservation: When first loaded to GPU, reserve INITIAL_GPU_PAGE_BUFFER pages
      beyond current context to reduce load/on-hold churning.
    - Extension reservation: At page boundaries, extend by EXTENSION_GPU_PAGE_BUFFER pages.
    - New KV tokens are streamed to host after each attention layer.
    - Sequences can be put ON_HOLD when GPU memory is insufficient.

    Configuration:
    Call configure_page_buffers() at startup to set page buffer values.
    Server flags:
    - --initial-gpu-page-buffer: Pages to reserve on first GPU load (default: 32)
    - --extension-gpu-page-buffer: Pages to add at boundaries (default: 4)
    - --decision-frequency-pages: How often to make scheduling decisions (default: 2)
      Must be <= extension_gpu_page_buffer to ensure sequences don't overflow between decisions.
"""
from enum import IntEnum
from typing import Dict, List, Optional, Set
import torch
import math


# Default page buffer sizes (can be overridden via configure_page_buffers())
# Initial reservation when sequence is first loaded to GPU (e.g., after prefill or from ON_HOLD)
INITIAL_GPU_PAGE_BUFFER = 32  # Pages to reserve on first GPU load
# Extension buffer at page boundaries (for sequences already in decode)
EXTENSION_GPU_PAGE_BUFFER = 4  # Pages to add at boundaries
# Decision frequency: how many pages (each 64 tokens) between boundary checks
DECISION_FREQUENCY_PAGES = 2  # How often to make scheduling decisions (in pages)


def configure_page_buffers(
    initial_gpu_page_buffer: int = 32,
    extension_gpu_page_buffer: int = 4,
    decision_frequency_pages: int = 2
) -> None:
    """Configure page buffer settings at runtime.

    This should be called during server/worker initialization with values
    from server args.

    Args:
        initial_gpu_page_buffer: Pages to reserve on first GPU load (default: 32)
        extension_gpu_page_buffer: Pages to add at page boundaries (default: 4)
        decision_frequency_pages: How often to make scheduling decisions in pages (default: 2)
    """
    global INITIAL_GPU_PAGE_BUFFER, EXTENSION_GPU_PAGE_BUFFER, DECISION_FREQUENCY_PAGES

    INITIAL_GPU_PAGE_BUFFER = initial_gpu_page_buffer
    EXTENSION_GPU_PAGE_BUFFER = extension_gpu_page_buffer
    DECISION_FREQUENCY_PAGES = decision_frequency_pages


class SequenceStatus(IntEnum):
    QUEUEING = 0      # Waiting to be prefilled
    IN_PREFILL = 1    # Currently being prefilled
    PREFILLED = 2     # Prefilled, KV in host, waiting for GPU
    IN_DECODE = 3     # Currently decoding with KV in GPU
    ON_HOLD = 4       # Was decoding, paused due to GPU memory pressure, KV in host
    COMPLETED = 5     # Finished generation
    EVICTED = 6       # Host KV fully released, only token IDs retained; awaits prefill recompute


class SequenceEntry:
    __slots__ = (
        'uuid', 'global_idx', 'prompt_length', 'max_decode_length',
        'status', 'decoded_length', 'current_context_length',
        'input_ids', 'decoded_tokens',
        'kv_token_budget', 'assigned_rank', 'text', 'eos_reached',
        # Two-page buffer tracking
        'gpu_pages_allocated',
        # Track whether this sequence has had its initial GPU reservation
        'had_initial_gpu_reservation',
        # Dynamic host KV reservation tracking
        'host_token_capacity',   # Current host KV capacity in tokens (grows by chunk)
        'host_pages_allocated',  # Current host page count
        # Eviction support
        'evicted_token_ids',     # Saved (prompt + decoded) tokens for recompute after eviction
        'original_prompt_length',  # Original prompt length before eviction (for tracking)
        'original_max_decode_length',  # Original max_decode_length before eviction
        'total_decoded_before_eviction',  # Tokens decoded before this eviction cycle
        # Baseline decoded_length at the most recent re-entry. Set at re-entry
        # prep to n_old (the count of previously-decoded tokens that were
        # restored into decoded_tokens buffer for output preservation). At the
        # NEXT eviction, only tokens decoded BEYOND this baseline are genuinely
        # new and should be appended to evicted_token_ids — the tokens at
        # positions [0:baseline] are already present in the reconstructed
        # prompt (input_ids) and appending them again would create a geometric
        # cascade of duplicated tokens across re-entry cycles.
        'reentry_decoded_baseline',
        '_buffer_slot',  # Index into QueryBookBufferPool buffers
        # Request pool fields
        'batch_id',              # Which batch this sequence belongs to (for result routing)
        'pool_slot_index',       # Index in SchedulingPool's pre-allocated QueryBook
        'priority',              # 0=NORMAL, 1=HIGH (inherited from batch)
        'sampling_params',       # Per-request sampling params for this sequence
        # Lifespan monitoring (BATCHGEN_SEQ_LIFESPAN=1)
        '_lifespan_log',   # List[SeqEventRecord], ring buffer
        '_lifespan_idx',   # int, next write position
        # Repetition detection
        '_rep_last_token',  # int, last token ID seen
        '_rep_count',       # int, consecutive same-token count
        '_rep_detected',    # bool, whether repetition was detected
    )

    VALID_TRANSITIONS = {
        SequenceStatus.QUEUEING: {SequenceStatus.IN_PREFILL},
        SequenceStatus.IN_PREFILL: {SequenceStatus.PREFILLED},
        SequenceStatus.PREFILLED: {SequenceStatus.IN_DECODE, SequenceStatus.COMPLETED},
        SequenceStatus.IN_DECODE: {SequenceStatus.ON_HOLD, SequenceStatus.COMPLETED, SequenceStatus.EVICTED},
        SequenceStatus.ON_HOLD: {SequenceStatus.IN_DECODE, SequenceStatus.COMPLETED, SequenceStatus.EVICTED},
        SequenceStatus.COMPLETED: set(),
        SequenceStatus.EVICTED: {SequenceStatus.IN_PREFILL},  # Re-enters via prefill recompute
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
        self.decoded_tokens: Optional[torch.Tensor] = None
        self.kv_token_budget: int = prompt_length + max_decode_length
        self.assigned_rank: Optional[int] = None
        self.text = text
        self.eos_reached: bool = False
        
        # Two-page buffer: tracks current GPU page allocation
        # This is separate from host pages (which always hold full kv_token_budget)
        self.gpu_pages_allocated: int = 0

        # Track whether this sequence has had its initial large GPU reservation
        # Reset to False when sequence goes ON_HOLD (GPU pages released)
        self.had_initial_gpu_reservation: bool = False

        # Dynamic host KV reservation: starts at 0, set by worker at prefill time
        self.host_token_capacity: int = 0
        self.host_pages_allocated: int = 0

        # Eviction support
        self.evicted_token_ids: Optional[torch.Tensor] = None
        self.original_prompt_length: int = prompt_length
        self.original_max_decode_length: int = max_decode_length
        self.total_decoded_before_eviction: int = 0
        # Baseline decoded_length at the most recent re-entry. Starts at 0 for
        # fresh sequences that have never been evicted. Incremented at each
        # re-entry prep so that subsequent evictions only append NEW decoded
        # tokens (not the ones already baked into the reconstructed prompt).
        self.reentry_decoded_baseline: int = 0
        self._buffer_slot: int = -1

        # Request pool fields (set externally when using pool-based scheduling)
        self.batch_id: Optional[str] = None
        self.pool_slot_index: int = -1
        self.priority: int = 0  # 0=NORMAL, 1=HIGH
        self.sampling_params: Optional[Dict] = None

        # Lifespan monitoring
        self._lifespan_log: list = []
        self._lifespan_idx: int = 0

        # Repetition detection
        self._rep_last_token: int = -1
        self._rep_count: int = 0
        self._rep_detected: bool = False

    def log_event(self, event: int, rank: int, detail: str = "") -> None:
        """Log a lifespan event. No-op when BATCHGEN_SEQ_LIFESPAN is not set."""
        import logging
        from batchgen.lifespan import ENABLED, MAX_EVENTS, SeqEvent, SeqEventRecord
        if not ENABLED:
            return
        expected_ctx = self.original_prompt_length + self.decoded_length
        record = SeqEventRecord(
            event=event,
            rank=rank,
            decoded_length=self.decoded_length,
            current_ctx=self.current_context_length,
            expected_ctx=expected_ctx,
            gpu_pages=self.gpu_pages_allocated,
            host_pages=self.host_pages_allocated,
            detail=detail,
        )
        if len(self._lifespan_log) < MAX_EVENTS:
            self._lifespan_log.append(record)
        else:
            self._lifespan_log[self._lifespan_idx % MAX_EVENTS] = record
        self._lifespan_idx += 1
        # Write to server log for immediate visibility
        try:
            name = SeqEvent(event).name
        except ValueError:
            name = f"EVT{event}"
        mismatch = " ***MISMATCH***" if self.current_context_length != expected_ctx else ""
        logging.info(
            f"[LIFESPAN] {self.uuid[:8]} gid={self.global_idx} {name} "
            f"rank={rank} dec={self.decoded_length} ctx={self.current_context_length} "
            f"exp={expected_ctx} gpu_pg={self.gpu_pages_allocated} "
            f"host_pg={self.host_pages_allocated} {detail}{mismatch}"
        )

    def validate_metadata(self, context: str, require_owner_tensors: bool = True) -> None:
        """Reject inconsistent per-sequence metadata at module boundaries."""
        prefix = f"{context}: sequence {self.uuid} gid={self.global_idx}"

        def require(condition: bool, message: str) -> None:
            if not condition:
                raise RuntimeError(f"{prefix}: {message}")

        require(self.global_idx >= 0, f"global_idx must be non-negative, got {self.global_idx}")
        require(self.prompt_length >= 0, f"prompt_length must be non-negative, got {self.prompt_length}")
        require(
            self.original_prompt_length >= 0,
            f"original_prompt_length must be non-negative, got {self.original_prompt_length}",
        )
        require(self.max_decode_length >= 0, f"max_decode_length must be non-negative, got {self.max_decode_length}")
        require(
            self.original_max_decode_length >= 0,
            f"original_max_decode_length must be non-negative, got {self.original_max_decode_length}",
        )
        require(self.decoded_length >= 0, f"decoded_length must be non-negative, got {self.decoded_length}")
        require(
            self.total_decoded_before_eviction >= 0,
            f"total_decoded_before_eviction must be non-negative, got {self.total_decoded_before_eviction}",
        )
        require(
            self.reentry_decoded_baseline >= 0,
            f"reentry_decoded_baseline must be non-negative, got {self.reentry_decoded_baseline}",
        )
        require(
            self.reentry_decoded_baseline <= self.decoded_length,
            f"reentry_decoded_baseline={self.reentry_decoded_baseline} exceeds decoded_length={self.decoded_length}",
        )
        require(
            self.total_decoded_before_eviction <= self.decoded_length,
            f"total_decoded_before_eviction={self.total_decoded_before_eviction} exceeds decoded_length={self.decoded_length}",
        )

        expected_ctx = self.original_prompt_length + self.decoded_length
        require(
            self.current_context_length == expected_ctx,
            f"current_context_length={self.current_context_length} must equal "
            f"original_prompt_length + decoded_length = {expected_ctx}",
        )

        expected_budget = self.original_prompt_length + self.original_max_decode_length
        require(
            self.kv_token_budget == expected_budget,
            f"kv_token_budget={self.kv_token_budget} must equal "
            f"original_prompt_length + original_max_decode_length = {expected_budget}",
        )

        require(self.host_pages_allocated >= 0, f"host_pages_allocated must be non-negative, got {self.host_pages_allocated}")
        require(self.host_token_capacity >= 0, f"host_token_capacity must be non-negative, got {self.host_token_capacity}")
        require(self.gpu_pages_allocated >= 0, f"gpu_pages_allocated must be non-negative, got {self.gpu_pages_allocated}")
        if self.host_pages_allocated:
            expected_host_capacity = self.host_pages_allocated * self.PAGE_SIZE
            require(
                self.host_token_capacity == expected_host_capacity,
                f"host_token_capacity={self.host_token_capacity} must equal "
                f"host_pages_allocated * PAGE_SIZE = {expected_host_capacity}",
            )

        host_required_statuses = {
            SequenceStatus.PREFILLED,
            SequenceStatus.IN_DECODE,
            SequenceStatus.ON_HOLD,
        }
        if self.status in host_required_statuses:
            require(self.assigned_rank is not None, f"{self.status.name} requires assigned_rank")
            require(self.host_pages_allocated > 0, f"{self.status.name} requires host_pages_allocated > 0")
            require(
                self.host_token_capacity >= self.current_context_length,
                f"host_token_capacity={self.host_token_capacity} is smaller than current_context_length={self.current_context_length}",
            )

        if self.status == SequenceStatus.IN_DECODE:
            require(self.gpu_pages_allocated > 0, "IN_DECODE requires gpu_pages_allocated > 0")
            require(
                self.gpu_pages_allocated * self.PAGE_SIZE >= self.current_context_length,
                f"gpu allocation tokens={self.gpu_pages_allocated * self.PAGE_SIZE} "
                f"is smaller than current_context_length={self.current_context_length}",
            )
        elif self.status == SequenceStatus.ON_HOLD:
            require(self.gpu_pages_allocated == 0, f"ON_HOLD requires gpu_pages_allocated=0, got {self.gpu_pages_allocated}")
        elif self.status == SequenceStatus.EVICTED:
            require(self.assigned_rank is not None, "EVICTED requires assigned_rank for deterministic re-entry")
            require(self.gpu_pages_allocated == 0, f"EVICTED requires gpu_pages_allocated=0, got {self.gpu_pages_allocated}")
            require(self.host_pages_allocated == 0, f"EVICTED requires host_pages_allocated=0, got {self.host_pages_allocated}")
            if require_owner_tensors:
                require(self.evicted_token_ids is not None, "EVICTED owner requires evicted_token_ids")
            else:
                require(
                    self.evicted_token_ids is None,
                    "non-owner EVICTED replica must not retain owner-only evicted_token_ids",
                )
                require(
                    self.total_decoded_before_eviction > 0,
                    "non-owner EVICTED replica requires total_decoded_before_eviction > 0",
                )

        if self.input_ids is not None:
            if self.input_ids.dim() == 1:
                input_capacity = int(self.input_ids.numel())
            elif self.input_ids.dim() == 2 and int(self.input_ids.shape[0]) == 1:
                # QueryBookBufferPool exposes per-sequence storage as a
                # singleton-batch view; any batch dimension >1 is invalid.
                input_capacity = int(self.input_ids.shape[1])
            else:
                require(
                    False,
                    f"input_ids must be a per-sequence tensor or singleton view, got shape={tuple(self.input_ids.shape)}",
                )
            require(
                input_capacity >= self.prompt_length,
                f"input_ids capacity={input_capacity} is smaller than prompt_length={self.prompt_length}",
            )
            require(
                input_capacity <= self.kv_token_budget,
                f"input_ids capacity={input_capacity} exceeds kv_token_budget={self.kv_token_budget}",
            )
        if self.decoded_tokens is not None:
            require(
                self.decoded_tokens.dim() >= 1,
                f"decoded_tokens must have at least one dimension, got shape={tuple(self.decoded_tokens.shape)}",
            )
            require(
                int(self.decoded_tokens.shape[-1]) >= self.decoded_length,
                f"decoded_tokens last dimension={int(self.decoded_tokens.shape[-1])} "
                f"is smaller than decoded_length={self.decoded_length}",
            )

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

    # ============ Configurable Page Buffer Methods (GPU KV) ============

    def get_gpu_pages_for_initial_load(self) -> int:
        """
        Get GPU pages needed for INITIAL load (first time loading to GPU).
        
        Uses INITIAL_GPU_PAGE_BUFFER (default 64 pages = 4096 tokens) to give
        sequences a larger runway and reduce load/on-hold churning.
        
        This is used when:
        - Sequence transitions from PREFILLED to IN_DECODE
        - Sequence transitions from ON_HOLD back to IN_DECODE
        
        Returns:
            Number of pages to allocate on GPU for initial load
        """
        buffer_tokens = self.current_context_length + INITIAL_GPU_PAGE_BUFFER * self.PAGE_SIZE
        # Cap at full budget (don't over-allocate beyond what sequence needs)
        buffer_tokens = min(buffer_tokens, self.kv_token_budget)
        return math.ceil(buffer_tokens / self.PAGE_SIZE)

    def get_gpu_pages_for_two_page_buffer(self) -> int:
        """
        Get GPU pages needed for page buffer design.

        For INITIAL load (had_initial_gpu_reservation=False):
            Allocates enough pages for: current_context_length + INITIAL_GPU_PAGE_BUFFER*PAGE_SIZE tokens.
            This gives sequences a large runway to reduce load/on-hold traffic.

        For EXTENSION (had_initial_gpu_reservation=True):
            Allocates enough pages for: current_context_length + EXTENSION_GPU_PAGE_BUFFER*PAGE_SIZE tokens.
            This ensures we have a small buffer before needing extension.

        Returns:
            Number of pages to allocate on GPU (capped at host_pages_allocated)
        """
        if not self.had_initial_gpu_reservation:
            # Initial load - use larger buffer
            buffer_pages = INITIAL_GPU_PAGE_BUFFER
        else:
            # Extension - use smaller buffer
            buffer_pages = EXTENSION_GPU_PAGE_BUFFER

        buffer_tokens = self.current_context_length + buffer_pages * self.PAGE_SIZE
        # Cap at full budget (don't over-allocate)
        buffer_tokens = min(buffer_tokens, self.kv_token_budget)
        pages = math.ceil(buffer_tokens / self.PAGE_SIZE)
        # Cap at host pages: GPU can't load more pages than host has allocated
        if self.host_pages_allocated > 0:
            pages = min(pages, self.host_pages_allocated)
        return pages

    def get_additional_gpu_pages_needed(self) -> int:
        """
        Get additional GPU pages needed to maintain page buffer.
        
        Called at page boundaries to check if we need to extend GPU allocation.
        Uses EXTENSION_GPU_PAGE_BUFFER for extension decisions.
        
        Returns:
            Number of additional pages needed (0 if current allocation is sufficient)
        """
        # For extension, always use the extension buffer size
        buffer_tokens = self.current_context_length + EXTENSION_GPU_PAGE_BUFFER * self.PAGE_SIZE
        buffer_tokens = min(buffer_tokens, self.kv_token_budget)
        required = math.ceil(buffer_tokens / self.PAGE_SIZE)
        return max(0, required - self.gpu_pages_allocated)

    def mark_initial_gpu_reservation_done(self) -> None:
        """
        Mark that this sequence has received its initial GPU reservation.
        Call this after the first GPU page allocation for this sequence.
        """
        self.had_initial_gpu_reservation = True

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
        Also resets had_initial_gpu_reservation so next load gets the large initial buffer.
        """
        self.gpu_pages_allocated = 0
        self.had_initial_gpu_reservation = False

    # ============ Dynamic Host KV Reservation Methods ============

    def needs_host_kv_growth(self, chunk_size: int) -> bool:
        """Check if sequence needs more host KV pages to continue decoding.

        Returns True when the sequence is approaching its current host token
        capacity, leaving enough runway for one extension buffer.
        """
        if self.host_token_capacity <= 0:
            return False
        # Trigger growth when within one extension buffer of capacity
        runway = self.host_token_capacity - self.current_context_length
        threshold = EXTENSION_GPU_PAGE_BUFFER * self.PAGE_SIZE
        return runway <= threshold

    def get_host_growth_pages(self, chunk_size: int) -> int:
        """Get number of pages to grow host KV by.

        Returns chunk_size / PAGE_SIZE pages, capped so that total host
        capacity does not exceed kv_token_budget.
        """
        remaining_budget = self.kv_token_budget - self.host_token_capacity
        if remaining_budget <= 0:
            return 0
        growth_tokens = min(chunk_size, remaining_budget)
        return math.ceil(growth_tokens / self.PAGE_SIZE)

    def get_host_pages_for_initial_chunk(self, chunk_size: int) -> int:
        """Get pages for initial host KV allocation.

        Must match the actual allocation in _config_prefill_host_kv():
        max(prompt + chunk_size, ceil((prompt+1)/PAGE_SIZE) + INITIAL_GPU_PAGE_BUFFER pages)
        The +1 accounts for the first decoded token produced during prefill
        (current_context_length = prompt_length + 1 after prefill).
        """
        chunk_tokens = self.prompt_length + chunk_size
        post_prefill_length = self.prompt_length + 1  # prefill produces 1 decode token
        gpu_initial_pages = math.ceil(post_prefill_length / self.PAGE_SIZE) + INITIAL_GPU_PAGE_BUFFER
        gpu_initial_tokens = gpu_initial_pages * self.PAGE_SIZE
        initial_tokens = max(chunk_tokens, gpu_initial_tokens)
        initial_tokens = min(initial_tokens, self.kv_token_budget)
        return math.ceil(initial_tokens / self.PAGE_SIZE)

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
            f"ctx_len={self.current_context_length}, gpu_pages={self.gpu_pages_allocated}, "
            f"host_cap={self.host_token_capacity}, host_pages={self.host_pages_allocated})"
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
        # Handle UUID collision: if a sequence with this uuid already exists,
        # clean it out of the status and rank indices first. Otherwise the old
        # status entry lingers (e.g., COMPLETED) even after the dict is
        # overwritten, corrupting all_completed() and status-based queries.
        # This matters for pool mode when multiple batches reuse request_ids
        # (e.g., mmlu-0..mmlu-31).
        existing = self.sequences.get(sequence.uuid)
        if existing is not None:
            self._status_index[existing.status].discard(sequence.uuid)
            if (existing.assigned_rank is not None
                    and existing.assigned_rank in self._rank_index):
                self._rank_index[existing.assigned_rank].discard(sequence.uuid)
        self.sequences[sequence.uuid] = sequence
        self._status_index[sequence.status].add(sequence.uuid)
        if sequence.assigned_rank is not None:
            if sequence.assigned_rank not in self._rank_index:
                self._rank_index[sequence.assigned_rank] = set()
            self._rank_index[sequence.assigned_rank].add(sequence.uuid)

    def remove_sequence(self, uuid: str) -> Optional[SequenceEntry]:
        """Remove a sequence from the batch and all indices."""
        seq = self.sequences.pop(uuid, None)
        if seq:
            self._status_index[seq.status].discard(uuid)
            if seq.assigned_rank is not None and seq.assigned_rank in self._rank_index:
                self._rank_index[seq.assigned_rank].discard(uuid)
        return seq

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
        return sorted(self._status_index[status])

    def get_sequences_for_rank(self, rank: int) -> List[str]:
        rank_seqs = self._rank_index.get(rank, set())
        return sorted(rank_seqs, key=lambda uuid: self.sequences[uuid].global_idx)

    def get_sequences_for_rank_with_status(
        self, rank: int, status: SequenceStatus
    ) -> List[str]:
        rank_seqs = self._rank_index.get(rank, set())
        status_seqs = self._status_index[status]
        return sorted(
            rank_seqs & status_seqs,
            key=lambda uuid: self.sequences[uuid].global_idx,
        )

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

    def has_evicted(self) -> bool:
        """Check if there are sequences evicted from host KV awaiting recompute."""
        return len(self._status_index[SequenceStatus.EVICTED]) > 0

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
