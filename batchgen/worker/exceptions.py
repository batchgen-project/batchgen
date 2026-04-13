"""Worker package exceptions.

Every exception here is deliberately *loud*: handlers raise instead of
silently repairing state. The previous refactor's bug tail came from
log-and-fix paths that hid real drift; the new contract (plan Decision #6)
forbids that pattern.
"""

from __future__ import annotations

from typing import Literal


class CtxInvariantViolation(RuntimeError):
    """Raised when the CTX invariant is violated during metadata sync.

    Invariant: ``seq.current_context_length == seq.original_prompt_length +
    seq.decoded_length``. Detected on both sides of
    `SyncCoordinator.sync_metadata`:

      - ``side="sender"``: a sequence owned by this rank has inconsistent
        metadata BEFORE the collective is issued. No collective is emitted.
      - ``side="receiver"``: a sequence received from another rank fails the
        check after the collective. The collective has already completed.

    Attributes:
        uuid: Sequence UUID whose CTX invariant is broken.
        side: Which side detected the violation ("sender" or "receiver").
        had: The value currently stored on ``current_context_length``.
        expected: The value the invariant demands
            (``original_prompt_length + decoded_length``).
    """

    def __init__(
        self,
        *,
        uuid: str,
        side: Literal["sender", "receiver"],
        had: int,
        expected: int,
    ) -> None:
        self.uuid = uuid
        self.side = side
        self.had = had
        self.expected = expected
        super().__init__(
            f"CTX invariant violated ({side}) for seq {uuid!r}: "
            f"current_context_length={had}, expected={expected} "
            f"(original_prompt_length + decoded_length)"
        )


class GpuKvExhaustion(RuntimeError):
    """Raised when a GPU KV allocation cannot fit in the free page budget.

    Phase 2.5 replaces the legacy "log ERROR + return False" fallback in
    `_allocate_gpu_kv_two_page_buffer` and
    `_allocate_and_load_gpu_kv_for_new_sequences` with this exception so
    pressure-relief bugs surface a real stack trace instead of silently
    dropping the in-flight batch.

    Attributes:
        rank: Rank that detected the exhaustion.
        needed: Number of GPU pages the allocation requested.
        free: Number of free GPU pages available at the check.
        num_sequences: How many sequences the allocation covered.
        site: Short label of the call site ("two_page_buffer" /
            "load_new_sequences") so catchers can disambiguate.
    """

    def __init__(
        self,
        *,
        rank: int,
        needed: int,
        free: int,
        num_sequences: int,
        site: str,
    ) -> None:
        self.rank = rank
        self.needed = needed
        self.free = free
        self.num_sequences = num_sequences
        self.site = site
        super().__init__(
            f"GPU KV exhaustion at {site} on rank {rank}: "
            f"need {needed} pages, only {free} free "
            f"({num_sequences} sequences)"
        )


__all__ = ["CtxInvariantViolation", "GpuKvExhaustion"]
