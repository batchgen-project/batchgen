"""CompletionHandler — EOS/length/repetition detection and completion reporting.

Responsibilities:
  - `is_completed(seq)`: combined check for max_decode_length, context_limit,
    EOS-with-ignore-gating, and repetition-detected.
  - `get_finish_reason(seq)`: ``"repetition" | "stop" | "length"`` — priority
    order matches the detection order so the callers always see the primary
    cause (plan Decision #4: repetition also sets eos_reached, so repetition
    MUST be checked before stop).
  - `should_stop_at_eos(seq, token_id)`: plural-EOS lookup via
    ``getattr(tokenizer, 'eos_token_ids', {eos})`` per `conventions.md`; the
    ``ignore_eos`` override is a constructor arg so handlers never touch
    ``os.environ`` (plan Decision #3).
  - `check_repeating_pattern(seq)`: N-gram self-repeat detection over the
    decoded-tokens tail. Sets ``seq._rep_detected`` AND ``seq.eos_reached``
    on first hit (plan Decision #4, immediate stop).
  - `gather_tokens(uuids)`: cross-rank `all_gather_object` collecting decoded
    text for each assigned uuid; merges back to a dict.
  - `report(uuid, text, finish_reason)`: forwards to
    `ResponseSinkBackend` (rank-0-only — caller guards).
  - `check_and_handle(uuids)`: orchestrates detection → gather → report →
    transition, returning the set of newly-completed UUIDs. Uses
    `state.global_batch.update_status` (which calls `status_transition()`)
    so the atomic-transition invariant from the plan is preserved.
"""

from __future__ import annotations

from typing import Any

from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker.protocols import (
    UUID,
    CollectiveBackend,
    ResponseSinkBackend,
    TokenizerBackend,
)
from batchgen.worker.state import WorkerState


# Repetition detection window (plan Decision #4, matches main's ca6a9b37 + 163083bf).
REP_DETECTION_MIN_DECODED = 64  # Start checking only after 64 tokens decoded.
REP_DETECTION_PATTERN_MIN = 2
REP_DETECTION_PATTERN_MAX = 100


class CompletionHandler:
    def __init__(
        self,
        state: WorkerState,
        tokenizer: TokenizerBackend,
        collectives: CollectiveBackend,
        sink: ResponseSinkBackend,
        *,
        model_context_length: int,
        ignore_eos: bool = False,
        rep_detection_enabled: bool = True,
    ) -> None:
        self._state = state
        self._tokenizer = tokenizer
        self._collectives = collectives
        self._sink = sink
        self._model_context_length = model_context_length
        self._ignore_eos = ignore_eos
        self._rep_detection_enabled = rep_detection_enabled

    # ------------------------------------------------------------------
    # Predicates
    # ------------------------------------------------------------------

    def is_completed(self, seq: SequenceEntry) -> bool:
        """True if any completion condition fires for `seq`.

        Conditions (any-of, order independent):
          - `_rep_detected`
          - `decoded_length >= max_decode_length`
          - `current_context_length >= model_context_length`
          - `eos_reached AND NOT ignore_eos`
        """
        if seq._rep_detected:
            return True
        if seq.decoded_length >= seq.max_decode_length:
            return True
        if seq.current_context_length >= self._model_context_length:
            return True
        if seq.eos_reached and not self._ignore_eos:
            return True
        return False

    def get_finish_reason(self, seq: SequenceEntry) -> str:
        """Return the reason string; priority: repetition > stop > length.

        Checks repetition first because plan Decision #4 also sets
        ``eos_reached=True`` on detection; without this priority the caller
        would get ``"stop"`` for a repetition-triggered completion.

        Raises `ValueError` if called on a non-completed sequence — forces
        callers to guard with `is_completed` first rather than silently
        returning a misleading reason.
        """
        if seq._rep_detected:
            return "repetition"
        if seq.eos_reached and not self._ignore_eos:
            return "stop"
        if seq.decoded_length >= seq.max_decode_length:
            return "length"
        if seq.current_context_length >= self._model_context_length:
            return "length"
        raise ValueError(
            f"get_finish_reason called on non-completed sequence {seq.uuid!r}"
        )

    def should_stop_at_eos(self, seq: SequenceEntry, token_id: int) -> bool:
        """True if `token_id` is in the tokenizer's plural EOS set.

        Honors `conventions.md`: uses
        ``getattr(tokenizer, 'eos_token_ids', {eos_token_id})`` so custom
        tokenizers with multi-token stop lists are respected. The
        ``ignore_eos`` constructor arg short-circuits the check.
        """
        if self._ignore_eos:
            return False
        eos_ids = getattr(self._tokenizer, "eos_token_ids", None)
        if not eos_ids:
            return False
        return token_id in eos_ids

    # ------------------------------------------------------------------
    # Repetition detection
    # ------------------------------------------------------------------

    def check_repeating_pattern(self, seq: SequenceEntry) -> bool:
        """Detect trailing-N-gram self-repeat and set `_rep_detected` + `eos_reached`.

        Scans pattern lengths in ``[2, min(100, decoded_length // 2)]`` and
        flags ``seq._rep_detected = True`` on the first exact repeat. Returns
        ``True`` if a pattern was detected on this call.

        Short-circuits (returns ``False``) when detection is disabled, when
        ``decoded_length < 64``, or when ``decoded_tokens`` is unset.
        """
        if not self._rep_detection_enabled:
            return False
        if seq.decoded_length < REP_DETECTION_MIN_DECODED:
            return False
        if seq.decoded_tokens is None:
            return False

        tokens = seq.decoded_tokens[: seq.decoded_length].tolist()
        n = len(tokens)
        max_pattern = min(REP_DETECTION_PATTERN_MAX, n // 2)
        for pattern_len in range(REP_DETECTION_PATTERN_MIN, max_pattern + 1):
            last = tokens[n - pattern_len : n]
            prev = tokens[n - 2 * pattern_len : n - pattern_len]
            if last == prev:
                seq._rep_detected = True
                seq.eos_reached = True  # plan Decision #4: immediate stop.
                return True
        return False

    # ------------------------------------------------------------------
    # Cross-rank token gathering and reporting
    # ------------------------------------------------------------------

    def gather_tokens(self, uuids: list[UUID]) -> dict[UUID, str]:
        """All-gather decoded text for each owned uuid, return merged dict.

        Each rank decodes only its assigned uuids (``seq.assigned_rank ==
        state.rank``), then `all_gather_object` merges per-rank dicts into one
        authoritative mapping. Non-owned uuids contribute nothing.
        """
        local: dict[UUID, str] = {}
        for uuid in uuids:
            seq = self._state.global_batch.get_sequence(uuid)
            if seq is None or seq.assigned_rank != self._state.rank:
                continue
            if seq.decoded_tokens is None or seq.decoded_length == 0:
                local[uuid] = ""
                continue
            ids = seq.decoded_tokens[: seq.decoded_length].tolist()
            local[uuid] = self._tokenizer.decode(ids)

        gathered: list[Any] = [None] * self._state.world_size
        self._collectives.all_gather_object(gathered, local)

        merged: dict[UUID, str] = {}
        for rank_result in gathered:
            if rank_result:
                merged.update(rank_result)
        return merged

    def report(self, uuid: UUID, text: str, finish_reason: str) -> None:
        """Forward a completed sequence to the response sink (rank-0-only caller)."""
        self._sink.put(uuid, {"text": text, "finish_reason": finish_reason})

    # ------------------------------------------------------------------
    # Composition: detect → gather → report → transition
    # ------------------------------------------------------------------

    def check_and_handle(self, uuids: list[UUID]) -> set[UUID]:
        """Run the full completion cycle on `uuids`, return newly-completed set.

        Only IN_DECODE / PREFILLED sequences are considered (matching main's
        guarded transition). Sequences in any other status are skipped so a
        stale `_rep_detected` flag on a QUEUEING re-entry cannot trigger an
        invalid transition.

        The orchestration order preserves plan invariant #9 (atomic
        transition, metadata BEFORE status change):
          1. Detect completions (pure read).
          2. Gather decoded text across ranks (collective).
          3. For each completed uuid: read finish reason, report on rank 0,
             transition status via `SequenceBatch.update_status` (which
             delegates to `status_transition` — never mutate `seq.status`
             directly).
        """
        to_complete: list[UUID] = []
        for uuid in uuids:
            seq = self._state.global_batch.get_sequence(uuid)
            if seq is None:
                continue
            if seq.status not in (SequenceStatus.IN_DECODE, SequenceStatus.PREFILLED):
                continue
            if self.is_completed(seq):
                to_complete.append(uuid)

        if not to_complete:
            return set()

        texts = self.gather_tokens(to_complete)

        completed: set[UUID] = set()
        for uuid in to_complete:
            seq = self._state.global_batch.get_sequence(uuid)
            if seq is None:
                continue
            reason = self.get_finish_reason(seq)
            if self._state.rank == 0:
                self.report(uuid, texts.get(uuid, ""), reason)
            self._state.global_batch.update_status(uuid, SequenceStatus.COMPLETED)
            completed.add(uuid)
        return completed


__all__ = ["CompletionHandler"]
