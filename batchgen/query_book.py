"""Bounded paged token storage for active QueryBook sequences.

The server supplies a fixed byte budget. Each sequence reserves the complete
page chain required by its own maximum token length at bind time. The book
never grows, relocates, or evicts pages while serving a sequence.
"""

from dataclasses import dataclass
import operator
from typing import Dict, MutableMapping, Optional, Sequence, Set, Tuple

import torch


class QueryBookCapacityError(RuntimeError):
    """A QueryBook operation exceeded the configured fixed capacity."""


@dataclass(frozen=True)
class QueryBookSlot:
    """Lengths for one stable sequence slot."""

    slot: int
    prompt_length: int
    token_length: int
    decoded_length: int
    max_tokens: int


@dataclass(frozen=True)
class QueryBookTurn:
    """Logical prompt/generated spans for one trajectory turn.

    ``prompt_start`` and ``generated_start`` index the immutable logical
    trajectory, while ``generated_length`` counts only newly sampled tokens.
    A KV re-entry turn can therefore replay an existing prompt prefix without
    making the cumulative generated count look larger.
    """

    turn_id: int
    prompt_start: int
    prompt_length: int
    generated_start: int
    generated_length: int

    @property
    def prompt_end(self) -> int:
        return self.prompt_start + self.prompt_length

    @property
    def generated_end(self) -> int:
        return self.generated_start + self.generated_length


class QueryBook:
    """Fixed-byte-budget paged token book.

    The token store is one int32 CPU tensor of pages. ``bind`` reserves all
    pages for a sequence's ``max_tokens`` immediately. Callers should pass the
    full trajectory limit (prompt tokens plus the maximum generated suffix),
    so a request either receives its complete reservation or remains queued.
    Appending a token uses the sequence's cached tail page and offset; the hot
    append path does not perform a page-table lookup.
    """

    TOKEN_BYTES = 4

    def __init__(
        self,
        capacity_bytes: int,
        page_tokens: int = 4096,
        storage: Optional[torch.Tensor] = None,
    ) -> None:
        capacity_bytes = self._positive_int(capacity_bytes, "capacity_bytes")
        page_tokens = self._positive_int(page_tokens, "page_tokens")
        page_bytes = page_tokens * self.TOKEN_BYTES
        if capacity_bytes < page_bytes:
            raise ValueError(
                f"capacity_bytes={capacity_bytes} is smaller than one page={page_bytes}"
            )

        self.capacity_bytes = int(capacity_bytes)
        self.page_tokens = int(page_tokens)
        self.page_bytes = page_bytes
        self.page_count = capacity_bytes // page_bytes
        if storage is None:
            # The server may construct the book under inference_mode while the
            # scheduler later writes it from ordinary Python.
            with torch.inference_mode(False):
                storage = torch.empty(
                    (self.page_count, self.page_tokens),
                    dtype=torch.int32,
                    device="cpu",
                )
        if (
            not isinstance(storage, torch.Tensor)
            or storage.dtype != torch.int32
            or storage.device.type != "cpu"
            or not storage.is_contiguous()
            or tuple(storage.shape) != (self.page_count, self.page_tokens)
        ):
            raise ValueError(
                "storage must be a contiguous CPU int32 tensor with shape "
                f"({self.page_count}, {self.page_tokens})"
            )
        self.storage = storage
        # Store free page ranges instead of one Python integer per page. A
        # 50-GiB pool with 64-token pages has over 200 million pages; the
        # allocator metadata must remain proportional to fragmentation, not
        # to the byte budget.
        self._free_extents: list[tuple[int, int]] = [(0, self.page_count)]
        self._records: Dict[int, QueryBookSlot] = {}
        self._turns: Dict[int, list[QueryBookTurn]] = {}
        self._page_ranges: Dict[int, tuple[int, int]] = {}
        self._tail_pages: Dict[int, int] = {}
        self._tail_offsets: Dict[int, int] = {}
        self._seq_to_slot: Dict[str, int] = {}
        self._slot_to_seq: Dict[int, str] = {}
        self._free_slots: list[int] = []
        self._next_slot = 0

    @property
    def memory_bytes(self) -> int:
        """Bytes occupied by the fixed token tensor."""

        return int(self.storage.numel() * self.storage.element_size())

    @property
    def capacity_tokens(self) -> int:
        """Logical token slots in the allocated pages."""

        return self.page_count * self.page_tokens

    @property
    def free_page_count(self) -> int:
        return sum(count for _, count in self._free_extents)

    @property
    def free_bytes(self) -> int:
        return self.free_page_count * self.page_bytes

    @property
    def largest_free_extent_pages(self) -> int:
        """Largest contiguous reservation currently available."""

        return max((count for _, count in self._free_extents), default=0)

    @property
    def active_count(self) -> int:
        return len(self._records)

    def pages_for(self, max_tokens: int) -> int:
        max_tokens = self._positive_int(max_tokens, "max_tokens")
        return (max_tokens + self.page_tokens - 1) // self.page_tokens

    def can_reserve(self, max_tokens: int) -> bool:
        """Return whether admission can reserve a complete token chain."""

        return self.pages_for(max_tokens) <= self.largest_free_extent_pages

    def can_reserve_batch(self, max_tokens: Sequence[int]) -> bool:
        """Return whether a whole candidate batch fits in free extents.

        The check mirrors the first-fit allocator without mutating it. An
        aggregate free-page check is insufficient after releases fragment the
        pool: every sequence needs one contiguous chain of pages.
        """

        extents = list(self._free_extents)
        for tokens in max_tokens:
            pages = self.pages_for(tokens)
            for index, (start, count) in enumerate(extents):
                if count < pages:
                    continue
                if count == pages:
                    extents.pop(index)
                else:
                    extents[index] = (start + pages, count - pages)
                break
            else:
                return False
        return True

    def reservation_total_capacity(self, max_tokens: int) -> int:
        """Return the immutable number of equal-size reservations in the pool.

        This is a property of the allocated token tensor and the reservation
        size.  It does not change when sequences bind or release pages, so it
        is the only QueryBook capacity that may size the server's
        ``SchedulingPool``.
        """

        return self.page_count // self.pages_for(max_tokens)

    def reservation_free_count(self, max_tokens: int) -> int:
        """Return the currently available equal-size reservations.

        The free count reflects allocator fragmentation and therefore changes
        after every bind or release.  It is telemetry/admission information;
        it is not the scheduler pool's total capacity.
        """

        pages = self.pages_for(max_tokens)
        return sum(count // pages for _, count in self._free_extents)

    def reservation_capacity(self, max_tokens: int) -> int:
        """Compatibility alias for :meth:`reservation_total_capacity`.

        New code must call ``reservation_total_capacity`` or
        ``reservation_free_count`` explicitly.  Keeping this alias avoids
        breaking older callers while making its historical ambiguity harmless:
        ``reservation_capacity`` now has the conventional immutable-capacity
        meaning and is never used for free-page telemetry.
        """

        return self.reservation_total_capacity(max_tokens)

    def bind_batch(
        self, sequence_ids: Sequence[str], max_tokens: Sequence[int]
    ) -> list[int]:
        """Atomically reserve complete page chains for a candidate batch.

        Admission must check the aggregate reservation before mutating the book.
        This prevents a too-large batch from binding a prefix of its sequences
        and leaving the scheduler with a partially admitted batch.
        """

        if len(sequence_ids) != len(max_tokens):
            raise ValueError(
                "sequence_ids/max_tokens length mismatch: "
                f"{len(sequence_ids)} != {len(max_tokens)}"
            )
        if len(set(sequence_ids)) != len(sequence_ids):
            raise ValueError("sequence_ids must be unique within a batch")
        if any(
            not isinstance(sequence_id, str) or not sequence_id
            for sequence_id in sequence_ids
        ):
            raise ValueError("sequence_id must be a non-empty string")
        if any(sequence_id in self._seq_to_slot for sequence_id in sequence_ids):
            raise ValueError("sequence_id already bound")
        required_pages = sum(self.pages_for(tokens) for tokens in max_tokens)
        if not self.can_reserve_batch(max_tokens):
            raise QueryBookCapacityError(
                "QueryBook token budget exhausted for batch: "
                f"need={required_pages * self.page_bytes} bytes, "
                f"free={self.free_bytes} bytes"
            )

        # Reserve each chain first and publish metadata only after every extent
        # has been found. The preflight above mirrors first-fit allocation, and
        # this rollback keeps the operation atomic if state changes between
        # validation and reservation.
        reservations: list[tuple[str, int, int, int]] = []
        try:
            for sequence_id, tokens in zip(sequence_ids, max_tokens):
                max_tokens_int = self._positive_int(tokens, "max_tokens")
                page_count = self.pages_for(max_tokens_int)
                page_start = self._take_extent(page_count)
                reservations.append((sequence_id, page_start, page_count, max_tokens_int))
        except BaseException:
            for _, page_start, page_count, _ in reversed(reservations):
                self._return_extent(page_start, page_count)
            raise

        slots = []
        for sequence_id, page_start, page_count, max_tokens_int in reservations:
            if self._free_slots:
                slot = self._free_slots.pop()
            else:
                slot = self._next_slot
                self._next_slot += 1
            self._seq_to_slot[sequence_id] = slot
            self._slot_to_seq[slot] = sequence_id
            self._page_ranges[slot] = (page_start, page_count)
            self._tail_pages[slot] = page_start
            self._tail_offsets[slot] = 0
            self._records[slot] = QueryBookSlot(
                slot, 0, 0, 0, max_tokens_int
            )
            self._turns[slot] = []
            slots.append(slot)
        return slots

    def bind(self, sequence_id: str, max_tokens: int) -> int:
        """Reserve the full page chain for one active sequence."""

        if not isinstance(sequence_id, str) or not sequence_id:
            raise ValueError("sequence_id must be a non-empty string")
        if sequence_id in self._seq_to_slot:
            raise ValueError(f"sequence_id already bound: {sequence_id}")

        max_tokens = self._positive_int(max_tokens, "max_tokens")
        page_count = self.pages_for(max_tokens)
        if page_count > self.largest_free_extent_pages:
            raise QueryBookCapacityError(
                f"QueryBook token budget exhausted: need={page_count * self.page_bytes} "
                f"bytes, free={self.free_bytes} bytes"
            )

        page_start = self._take_extent(page_count)
        if self._free_slots:
            slot = self._free_slots.pop()
        else:
            slot = self._next_slot
            self._next_slot += 1
        self._seq_to_slot[sequence_id] = slot
        self._slot_to_seq[slot] = sequence_id
        self._page_ranges[slot] = (page_start, page_count)
        self._tail_pages[slot] = page_start
        self._tail_offsets[slot] = 0
        self._records[slot] = QueryBookSlot(slot, 0, 0, 0, max_tokens)
        self._turns[slot] = []
        return slot

    def release(self, slot: int) -> None:
        """Release all reserved pages after a sequence has completed."""

        self._check_active_slot(slot)
        sequence_id = self._slot_to_seq.pop(slot)
        del self._seq_to_slot[sequence_id]
        page_start, page_count = self._page_ranges.pop(slot)
        self._return_extent(page_start, page_count)
        del self._tail_pages[slot]
        del self._tail_offsets[slot]
        del self._records[slot]
        del self._turns[slot]
        self._free_slots.append(slot)

    def slot_for(self, sequence_id: str) -> int:
        try:
            return self._seq_to_slot[sequence_id]
        except KeyError as exc:
            raise KeyError(f"sequence is not active: {sequence_id}") from exc

    def metadata(self, slot: int) -> QueryBookSlot:
        self._check_active_slot(slot)
        return self._records[slot]

    def has_slot(self, slot: int) -> bool:
        """Return whether ``slot`` is currently reserved."""

        return isinstance(slot, int) and slot in self._records

    def has_sequence(self, sequence_id: str) -> bool:
        """Return whether a sequence owns a reservation in this local book."""

        return sequence_id in self._seq_to_slot

    def turns(self, slot: int) -> tuple[QueryBookTurn, ...]:
        """Return an immutable snapshot of the slot's turn ledger."""

        self._check_active_slot(slot)
        return tuple(self._turns[slot])

    def write_prompt(self, slot: int, prompt_tokens: torch.Tensor) -> None:
        """Copy one prompt into the reserved pages and reset decode length."""

        self._check_active_slot(slot)
        tokens = self._prepare_tokens(prompt_tokens)
        self._check_length(slot, int(tokens.numel()))
        self._write_span(slot, 0, tokens)
        record = self._records[slot]
        self._records[slot] = QueryBookSlot(
            slot, int(tokens.numel()), int(tokens.numel()), 0, record.max_tokens
        )
        self._update_tail(slot, int(tokens.numel()))
        self._turns[slot] = [
            QueryBookTurn(
                turn_id=0,
                prompt_start=0,
                prompt_length=int(tokens.numel()),
                generated_start=int(tokens.numel()),
                generated_length=0,
            )
        ]

    def append_prompt(self, slot: int, prompt_tokens: torch.Tensor) -> QueryBookTurn:
        """Append a later user prompt to an existing trajectory.

        Unlike ``write_prompt`` this never rewinds the tail. It creates a new
        turn whose generated span starts after the appended prompt.
        """

        self._check_active_slot(slot)
        tokens = self._prepare_tokens(prompt_tokens)
        record = self._records[slot]
        prompt_start = record.token_length
        new_length = prompt_start + int(tokens.numel())
        self._check_length(slot, new_length)
        self._write_span(slot, prompt_start, tokens)
        self._records[slot] = QueryBookSlot(
            slot,
            new_length,
            new_length,
            record.decoded_length,
            record.max_tokens,
        )
        self._update_tail(slot, new_length)
        turn = QueryBookTurn(
            turn_id=len(self._turns[slot]),
            prompt_start=prompt_start,
            prompt_length=int(tokens.numel()),
            generated_start=new_length,
            generated_length=0,
        )
        self._turns[slot].append(turn)
        return turn

    def begin_reentry_turn(self, slot: int, prompt_length: int) -> QueryBookTurn:
        """Record a KV re-entry prompt without copying or appending tokens.

        The effective prompt is an existing trajectory prefix. This method is
        idempotent when the same re-entry boundary is observed twice.
        """

        self._check_active_slot(slot)
        prompt_length = self._nonnegative_int(prompt_length, "prompt_length")
        record = self._records[slot]
        if prompt_length > record.token_length:
            raise QueryBookCapacityError(
                f"re-entry prompt_length={prompt_length} exceeds token_length={record.token_length}"
            )
        turns = self._turns[slot]
        if turns:
            last = turns[-1]
            if last.generated_length == 0 and last.generated_start == record.token_length:
                # Repeated eviction/re-entry bookkeeping before the next
                # generated token does not need one Python object per retry.
                updated = QueryBookTurn(
                    turn_id=last.turn_id,
                    prompt_start=0,
                    prompt_length=prompt_length,
                    generated_start=record.token_length,
                    generated_length=0,
                )
                turns[-1] = updated
                self._records[slot] = QueryBookSlot(
                    slot,
                    prompt_length,
                    record.token_length,
                    record.decoded_length,
                    record.max_tokens,
                )
                return updated
        turn = QueryBookTurn(
            turn_id=len(turns),
            prompt_start=0,
            prompt_length=prompt_length,
            generated_start=record.token_length,
            generated_length=0,
        )
        turns.append(turn)
        self._records[slot] = QueryBookSlot(
            slot,
            prompt_length,
            record.token_length,
            record.decoded_length,
            record.max_tokens,
        )
        return turn

    def set_prompt_length(self, slot: int, prompt_length: int) -> None:
        """Update the effective prefill prompt boundary without copying data."""

        self._check_active_slot(slot)
        prompt_length = self._nonnegative_int(prompt_length, "prompt_length")
        record = self._records[slot]
        if prompt_length > record.token_length:
            raise QueryBookCapacityError(
                f"prompt_length={prompt_length} exceeds token_length={record.token_length}"
            )
        self._records[slot] = QueryBookSlot(
            slot,
            prompt_length,
            record.token_length,
            record.decoded_length,
            record.max_tokens,
        )

    def synchronize_metadata(
        self,
        slot: int,
        *,
        prompt_length: int,
        token_length: int,
        decoded_length: int,
    ) -> None:
        """Refresh process-local lengths for bytes written by another rank.

        The token pages are node-shared, but allocator metadata is local to each
        process.  A rank that did not run a sequence's decode step therefore
        needs the owner's lengths before it can record the next re-entry turn.
        This updates only metadata and the cached tail; it never copies token
        bytes or allocates storage.
        """

        self._check_active_slot(slot)
        prompt_length = self._nonnegative_int(prompt_length, "prompt_length")
        token_length = self._nonnegative_int(token_length, "token_length")
        decoded_length = self._nonnegative_int(decoded_length, "decoded_length")
        if prompt_length > token_length:
            raise QueryBookCapacityError(
                f"prompt_length={prompt_length} exceeds token_length={token_length}"
            )
        self._check_length(slot, token_length)

        record = self._records[slot]
        turns = list(self._turns[slot])
        if turns:
            generated_before_last = sum(
                turn.generated_length for turn in turns[:-1]
            )
            last_generated = decoded_length - generated_before_last
            if last_generated < 0:
                raise QueryBookCapacityError(
                    "decoded_length is smaller than the committed turn ledger"
                )
            last = turns[-1]
            turns[-1] = QueryBookTurn(
                turn_id=last.turn_id,
                prompt_start=last.prompt_start,
                prompt_length=last.prompt_length,
                generated_start=last.generated_start,
                generated_length=last_generated,
            )
        elif token_length or decoded_length:
            # This is only a defensive fallback for a restored mirror whose
            # local turn list was empty.  Normal admission always creates the
            # first turn in write_prompt().
            generated_length = token_length - prompt_length
            if generated_length < 0 or generated_length != decoded_length:
                raise QueryBookCapacityError(
                    "cannot reconstruct an empty turn ledger from lengths"
                )
            turns = [
                QueryBookTurn(
                    turn_id=0,
                    prompt_start=0,
                    prompt_length=prompt_length,
                    generated_start=prompt_length,
                    generated_length=generated_length,
                )
            ]

        self._records[slot] = QueryBookSlot(
            slot,
            prompt_length,
            token_length,
            decoded_length,
            record.max_tokens,
        )
        self._turns[slot] = turns
        self._update_tail(slot, token_length)

    def write_prompts(
        self, slots: Sequence[int], prompts: Sequence[torch.Tensor]
    ) -> None:
        """Copy a batch of already-tokenized prompts into reserved chains."""

        if len(slots) != len(prompts):
            raise ValueError(
                f"slots/prompts length mismatch: {len(slots)} != {len(prompts)}"
            )
        if len(set(slots)) != len(slots):
            raise ValueError("slots must be unique within a batch")
        prepared = []
        for slot, prompt in zip(slots, prompts):
            self._check_active_slot(slot)
            tokens = self._prepare_tokens(prompt)
            self._check_length(slot, int(tokens.numel()))
            prepared.append(tokens)
        for slot, tokens in zip(slots, prepared):
            self._write_span(slot, 0, tokens)
            record = self._records[slot]
            prompt_length = int(tokens.numel())
            self._records[slot] = QueryBookSlot(
                slot, prompt_length, prompt_length, 0, record.max_tokens
            )
            self._update_tail(slot, prompt_length)
            self._turns[slot] = [
                QueryBookTurn(
                    turn_id=0,
                    prompt_start=0,
                    prompt_length=prompt_length,
                    generated_start=prompt_length,
                    generated_length=0,
                )
            ]

    def restore(
        self,
        slot: int,
        tokens: torch.Tensor,
        *,
        prompt_length: int,
        decoded_length: int,
        turns: Optional[Sequence[QueryBookTurn]] = None,
    ) -> None:
        """Restore an existing trajectory after rank/node migration.

        ``tokens`` contains the complete trajectory currently known to the
        sequence. ``prompt_length`` is the effective prompt for the next
        prefill turn; it may include earlier decoded tokens after KV eviction.
        ``decoded_length`` is cumulative generated-token bookkeeping, so it is
        intentionally independent of ``token_length - prompt_length``.
        """

        self._check_active_slot(slot)
        prompt_length = self._nonnegative_int(prompt_length, "prompt_length")
        decoded_length = self._nonnegative_int(decoded_length, "decoded_length")
        prepared = self._prepare_tokens(tokens)
        token_length = int(prepared.numel())
        if prompt_length > token_length:
            raise ValueError(
                f"prompt_length={prompt_length} exceeds token_length={token_length}"
            )
        self._check_length(slot, token_length)
        validated_turns = self._validate_turns(
            turns,
            token_length=token_length,
            prompt_length=prompt_length,
            decoded_length=decoded_length,
        )
        self._write_span(slot, 0, prepared)
        record = self._records[slot]
        self._records[slot] = QueryBookSlot(
            slot, prompt_length, token_length, decoded_length, record.max_tokens
        )
        self._update_tail(slot, token_length)
        self._turns[slot] = validated_turns

    def append_token(self, slot: int, token_id: int) -> int:
        """Append one generated token and return its logical token index."""

        self._check_active_slot(slot)
        record = self._records[slot]
        index = record.token_length
        self._check_length(slot, index + 1)
        page = self._tail_pages[slot]
        offset = self._tail_offsets[slot]
        self.storage[page, offset] = int(token_id)
        self._records[slot] = QueryBookSlot(
            slot,
            record.prompt_length,
            index + 1,
            record.decoded_length + 1,
            record.max_tokens,
        )
        self._update_tail(slot, index + 1)
        self._increment_last_turn(slot, index + 1)
        return index

    def append_tokens(
        self, slots: Sequence[int], token_ids: Sequence[int]
    ) -> None:
        """Append one token to every row in a decode microbatch."""

        if len(slots) != len(token_ids):
            raise ValueError(
                f"slots/token_ids length mismatch: {len(slots)} != {len(token_ids)}"
            )
        if len(set(slots)) != len(slots):
            raise ValueError("slots must be unique within a batch")
        prepared = []
        for slot, token_id in zip(slots, token_ids):
            self._check_active_slot(slot)
            record = self._records[slot]
            self._check_length(slot, record.token_length + 1)
            prepared.append(int(token_id))
        for slot, token_id in zip(slots, prepared):
            self.append_token(slot, token_id)

    def tokens(self, slot: int, length: Optional[int] = None) -> torch.Tensor:
        """Materialize a contiguous CPU tensor for a valid token span."""

        self._check_active_slot(slot)
        valid_length = self._records[slot].token_length
        if length is None:
            length = valid_length
        else:
            length = self._nonnegative_int(length, "length")
        if length < 0 or length > valid_length:
            raise QueryBookCapacityError(
                f"requested length={length}, valid length={valid_length}"
            )
        output = torch.empty(length, dtype=torch.int32)
        self.copy_span_to(slot, output, length=length)
        return output

    def generated_tokens(self, slot: int) -> torch.Tensor:
        """Materialize only generated spans, excluding later user prompts."""

        self._check_active_slot(slot)
        spans = [
            (turn.generated_start, turn.generated_length)
            for turn in self._turns[slot]
            if turn.generated_length
        ]
        if not spans:
            return torch.empty(0, dtype=torch.int32)
        output = torch.empty(sum(length for _, length in spans), dtype=torch.int32)
        offset = 0
        for start, length in spans:
            self._copy_span_to(slot, start, length, output[offset:offset + length])
            offset += length
        return output

    def last_generated_token(self, slot: int) -> torch.Tensor:
        """Materialize the latest generated token without assuming one suffix.

        A later prompt can sit between two generated spans, so callers must
        use the turn ledger instead of deriving the position from the original
        prompt length and cumulative decode count.
        """

        self._check_active_slot(slot)
        turns = self._turns[slot]
        for turn in reversed(turns):
            if turn.generated_length:
                output = torch.empty(1, dtype=torch.int32)
                self._copy_span_to(
                    slot, turn.generated_end - 1, 1, output
                )
                return output
        record = self._records[slot]
        if record.prompt_length <= 0:
            raise QueryBookCapacityError("sequence has no token to feed to decode")
        output = torch.empty(1, dtype=torch.int32)
        self._copy_span_to(slot, record.prompt_length - 1, 1, output)
        return output

    def turns_payload(self, slot: int) -> torch.Tensor:
        """Encode the turn ledger for a CPU migration payload."""

        turns = self.turns(slot)
        if not turns:
            return torch.empty((0, 6), dtype=torch.int64)
        return torch.tensor(
            [
                [
                    turn.turn_id,
                    turn.prompt_start,
                    turn.prompt_length,
                    turn.generated_start,
                    turn.generated_length,
                    turn.generated_end,
                ]
                for turn in turns
            ],
            dtype=torch.int64,
        )

    @staticmethod
    def turns_from_payload(payload: torch.Tensor) -> list[QueryBookTurn]:
        """Decode and validate the fixed-width migration turn payload."""

        if not isinstance(payload, torch.Tensor):
            raise ValueError("turn payload must be a tensor")
        if payload.dtype != torch.int64 or payload.device.type != "cpu":
            raise ValueError("turn payload must be a CPU int64 tensor")
        if payload.ndim != 2 or payload.shape[1] != 6:
            raise ValueError("turn payload must have shape [turns, 6]")
        turns = []
        for row in payload.tolist():
            (
                turn_id,
                prompt_start,
                prompt_length,
                generated_start,
                generated_length,
                generated_end,
            ) = row
            if generated_end != generated_start + generated_length:
                raise ValueError("turn payload generated_end is inconsistent")
            turns.append(
                QueryBookTurn(
                    turn_id=int(turn_id),
                    prompt_start=int(prompt_start),
                    prompt_length=int(prompt_length),
                    generated_start=int(generated_start),
                    generated_length=int(generated_length),
                )
            )
        return turns

    def copy_span_to(
        self,
        slot: int,
        output: torch.Tensor,
        *,
        start: int = 0,
        length: Optional[int] = None,
    ) -> torch.Tensor:
        """Copy a logical token span into a caller-owned contiguous tensor."""

        self._check_active_slot(slot)
        valid_length = self._records[slot].token_length
        start = self._nonnegative_int(start, "start")
        if length is None:
            length = valid_length - start
        else:
            length = self._nonnegative_int(length, "length")
        if start < 0 or length < 0 or start + length > valid_length:
            raise QueryBookCapacityError(
                f"requested span [{start}, {start + length}) exceeds valid length={valid_length}"
            )
        if (
            output.numel() < length
            or output.dtype != torch.int32
            or output.device.type != "cpu"
            or not output.is_contiguous()
        ):
            raise ValueError("output must be a sufficiently large CPU int32 tensor")
        flat_output = output.reshape(-1)
        self._copy_span_to(slot, start, length, flat_output)
        return flat_output[:length]

    def copy_to(
        self,
        slot: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.int64,
        length: Optional[int] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Copy a valid token span to a model input tensor."""

        self._check_active_slot(slot)
        valid_length = self._records[slot].token_length
        if length is None:
            length = valid_length
        else:
            length = self._nonnegative_int(length, "length")
        if length < 0 or length > valid_length:
            raise QueryBookCapacityError(
                f"requested length={length}, valid length={valid_length}"
            )
        target_device = torch.device(device)
        if target_device.type == "cuda" and target_device.index is None:
            target_device = torch.device("cuda", torch.cuda.current_device())
        source = torch.empty(length, dtype=torch.int32)
        self._copy_span_to(slot, 0, length, source)
        if out is not None:
            if (
                out.numel() < length
                or out.dtype != dtype
                or out.device != target_device
                or not out.is_contiguous()
            ):
                raise ValueError("out has incompatible shape, dtype, or device")
            # copy_ performs the dtype conversion and device transfer in one
            # operation; do not create a temporary GPU int64 tensor here.
            flat_out = out.reshape(-1)
            flat_out[:length].copy_(source)
            return flat_out[:length]
        return source.to(device=target_device, dtype=dtype)

    def _check_active_slot(self, slot: int) -> None:
        if not isinstance(slot, int) or slot not in self._records:
            raise QueryBookCapacityError(f"slot is not active: {slot}")

    @staticmethod
    def _nonnegative_int(value: int, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        try:
            value = operator.index(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be an integer, got {value!r}") from exc
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
        return value

    @staticmethod
    def _positive_int(value: int, name: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be an integer, got {value!r}")
        try:
            value = operator.index(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be an integer, got {value!r}") from exc
        if value <= 0:
            raise ValueError(f"{name} must be > 0, got {value}")
        return value

    @staticmethod
    def _prepare_tokens(tokens: torch.Tensor) -> torch.Tensor:
        if not isinstance(tokens, torch.Tensor):
            raise TypeError("tokens must be a torch.Tensor")
        return tokens.reshape(-1).to(dtype=torch.int32, device="cpu")

    def _check_length(self, slot: int, length: int) -> None:
        max_tokens = self._records[slot].max_tokens
        if length > max_tokens:
            raise QueryBookCapacityError(
                f"token length {length} exceeds reserved max_tokens={max_tokens}"
            )

    def _increment_last_turn(self, slot: int, token_end: int) -> None:
        turns = self._turns[slot]
        if not turns:
            record = self._records[slot]
            turns.append(
                QueryBookTurn(
                    turn_id=0,
                    prompt_start=0,
                    prompt_length=record.prompt_length,
                    generated_start=token_end - 1,
                    generated_length=0,
                )
            )
        last = turns[-1]
        turns[-1] = QueryBookTurn(
            turn_id=last.turn_id,
            prompt_start=last.prompt_start,
            prompt_length=last.prompt_length,
            generated_start=last.generated_start,
            generated_length=last.generated_length + 1,
        )

    @staticmethod
    def _validate_turns(
        turns: Optional[Sequence[QueryBookTurn]],
        *,
        token_length: int,
        prompt_length: int,
        decoded_length: int,
    ) -> list[QueryBookTurn]:
        if turns is None:
            generated_length = max(0, token_length - prompt_length)
            if generated_length != decoded_length:
                raise ValueError(
                    "migration restore requires turn metadata when cumulative "
                    "decoded length differs from the current prompt suffix"
                )
            fallback = QueryBookTurn(
                turn_id=0,
                prompt_start=0,
                prompt_length=prompt_length,
                generated_start=prompt_length,
                generated_length=generated_length,
            )
            return [fallback]
        validated = list(turns)
        generated_total = 0
        previous_generated_end = 0
        for expected_id, turn in enumerate(validated):
            if not isinstance(turn, QueryBookTurn):
                raise ValueError("turn ledger entries must be QueryBookTurn")
            if turn.turn_id != expected_id:
                raise ValueError("turn ids must be contiguous from zero")
            if min(
                turn.prompt_start,
                turn.prompt_length,
                turn.generated_start,
                turn.generated_length,
            ) < 0:
                raise ValueError("turn spans must be non-negative")
            if turn.prompt_end > token_length or turn.generated_end > token_length:
                raise ValueError("turn span exceeds restored trajectory length")
            if turn.prompt_end > turn.generated_start:
                raise ValueError("generated span overlaps its prompt span")
            if turn.generated_start < previous_generated_end:
                raise ValueError("generated spans must be monotonic and non-overlapping")
            generated_total += turn.generated_length
            previous_generated_end = turn.generated_end
        if not validated and (token_length or decoded_length):
            raise ValueError("non-empty trajectory requires at least one turn")
        if generated_total != decoded_length:
            raise ValueError(
                f"turn generated total={generated_total} does not match decoded_length={decoded_length}"
            )
        return validated

    def _update_tail(self, slot: int, logical_length: int) -> None:
        page_index, offset = divmod(logical_length, self.page_tokens)
        page_start, page_count = self._page_ranges[slot]
        if page_index >= page_count:
            page_index = page_count - 1
            offset = self.page_tokens
        self._tail_pages[slot] = page_start + page_index
        self._tail_offsets[slot] = offset

    def _write_span(self, slot: int, start: int, tokens: torch.Tensor) -> None:
        source = tokens.reshape(-1).to(dtype=torch.int32, device="cpu")
        remaining = int(source.numel())
        source_offset = 0
        logical = start
        while remaining:
            page_index, page_offset = divmod(logical, self.page_tokens)
            count = min(remaining, self.page_tokens - page_offset)
            page_start, _ = self._page_ranges[slot]
            page = self.storage[page_start + page_index]
            page[page_offset : page_offset + count].copy_(
                source[source_offset : source_offset + count]
            )
            logical += count
            source_offset += count
            remaining -= count

    def _copy_span_to(
        self, slot: int, start: int, length: int, output: torch.Tensor
    ) -> None:
        remaining = length
        output_offset = 0
        logical = start
        while remaining:
            page_index, page_offset = divmod(logical, self.page_tokens)
            count = min(remaining, self.page_tokens - page_offset)
            page_start, _ = self._page_ranges[slot]
            page = self.storage[page_start + page_index]
            output[output_offset : output_offset + count].copy_(
                page[page_offset : page_offset + count]
            )
            logical += count
            output_offset += count
            remaining -= count

    def _take_extent(self, page_count: int) -> int:
        """Take one contiguous free extent and return its first page."""

        for index, (start, count) in enumerate(self._free_extents):
            if count < page_count:
                continue
            if count == page_count:
                self._free_extents.pop(index)
            else:
                self._free_extents[index] = (start + page_count, count - page_count)
            return start
        raise QueryBookCapacityError(
            f"no contiguous extent of {page_count} pages; "
            f"free_pages={self.free_page_count}"
        )

    def _return_extent(self, start: int, count: int) -> None:
        """Return an extent and coalesce adjacent free ranges."""

        if count <= 0:
            return
        self._free_extents.append((start, count))
        self._free_extents.sort()
        merged: list[tuple[int, int]] = []
        for current_start, current_count in self._free_extents:
            if not merged or merged[-1][0] + merged[-1][1] < current_start:
                merged.append((current_start, current_count))
                continue
            previous_start, previous_count = merged[-1]
            previous_end = previous_start + previous_count
            merged[-1] = (
                previous_start,
                max(previous_end, current_start + current_count) - previous_start,
            )
        self._free_extents = merged


@dataclass
class QueryBookEntry:
    text: Optional[str] = None
    encoded: Optional[Dict[str, torch.Tensor]] = None
    decoded_tokens: Optional[torch.Tensor] = None
    kv_token_budget: Optional[int] = None


def make_query_book_entry(sequence) -> QueryBookEntry:
    return QueryBookEntry(
        text=sequence.text,
        encoded={"input_ids": sequence.input_ids},
        decoded_tokens=sequence.decoded_tokens,
        kv_token_budget=sequence.kv_token_budget,
    )


def bind_local_sequence_to_query_book(
    uuid: str,
    sequence,
    *,
    query_book: MutableMapping[int, QueryBookEntry],
    local_to_uuid_map: MutableMapping[int, str],
    uuid_to_local_map: MutableMapping[str, int],
    free_local_indices: Set[int],
    next_local_idx: int,
    local_idx: Optional[int] = None,
) -> Tuple[int, int]:
    if sequence is None:
        raise KeyError(f"Sequence with UUID {uuid} not found in global_batch")

    if local_idx is None:
        if free_local_indices:
            local_idx = free_local_indices.pop()
        else:
            local_idx = next_local_idx
            next_local_idx += 1

    local_to_uuid_map[local_idx] = uuid
    uuid_to_local_map[uuid] = local_idx
    query_book[local_idx] = make_query_book_entry(sequence)
    return local_idx, next_local_idx


def release_local_query_slot(
    uuid: str,
    *,
    uuid_to_local_map: MutableMapping[str, int],
    local_to_uuid_map: MutableMapping[int, str],
    query_book: MutableMapping[int, QueryBookEntry],
    free_local_indices: Set[int],
) -> Optional[int]:
    local_idx = uuid_to_local_map.pop(uuid, None)
    if local_idx is None:
        return None

    local_to_uuid_map.pop(local_idx, None)
    query_book.pop(local_idx, None)
    free_local_indices.add(local_idx)
    return local_idx
