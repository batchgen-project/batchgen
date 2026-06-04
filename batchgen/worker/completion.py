"""Sequence-completion detection helpers.

Slice 2 of the worker decouple initiative (issue #172). Ports the three
pure-predicate completion-detection methods previously inlined on
``BatchGenWorker`` (``_should_stop_at_eos``, ``_is_sequence_completed``,
``_get_finish_reason``) into a single sibling module.

Design follows the Phase A/B/C cuda-graph adapter pattern: a frozen
``CompletionContext`` snapshot carries the four worker-owned fields each
call consumes, and ``CompletionHandler`` is a namespace of stateless
static methods. The worker remains the canonical owner of the underlying
configuration; ``CompletionContext`` is a typed view, not a state copy.

Out of scope for this slice: ``_check_and_handle_completions`` (vectorized
boundary-time decision + repetition-flag mutation) and the cross-rank
``_sync_completion_status_*`` collectives — those land in later slices
(BatchFormation / SyncCoordinator).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, FrozenSet

from batchgen import lifespan
from batchgen.lifespan import SeqEvent

if TYPE_CHECKING:
    from batchgen.sequence import SequenceEntry


@dataclass(frozen=True)
class CompletionContext:
    """Frozen snapshot passed to each ``CompletionHandler`` call.

    The worker constructs one from its canonical fields (``self._ignore_eos``,
    ``self.eos_token_ids``, ``self.model_context_length``, ``self.rank``)
    per call site. Handler methods MUST NOT mutate any contents.
    """

    ignore_eos: bool
    eos_token_ids: FrozenSet[int]
    model_context_length: int
    rank: int


class CompletionHandler:
    """Namespace of stateless completion-detection predicates."""

    @staticmethod
    def should_stop_at_eos(ctx: CompletionContext, token_id: int) -> bool:
        """Return ``True`` iff this token is an EOS id and ``ignore_eos`` is off."""
        if ctx.ignore_eos:
            return False
        return token_id in ctx.eos_token_ids

    @staticmethod
    def is_sequence_completed(ctx: CompletionContext, seq: "SequenceEntry") -> bool:
        """Unified completion check that respects ``ignore_eos``.

        A sequence is completed if:
        1. ``decoded_length >= max_decode_length`` (always checked), OR
        2. ``current_context_length >= model_context_length`` (context limit), OR
        3. ``eos_reached and not ignore_eos`` (real EOS), OR
        4. Repetition pattern was detected on this sequence.
        """
        if seq.decoded_length >= seq.max_decode_length:
            return True
        if seq.current_context_length >= ctx.model_context_length:
            return True
        if seq.eos_reached and not ctx.ignore_eos:
            return True
        if seq._rep_detected:
            return True
        return False

    @staticmethod
    def get_finish_reason(ctx: CompletionContext, seq: "SequenceEntry") -> str:
        """Return OpenAI-compatible ``finish_reason`` for a completed sequence.

        ``seq.eos_reached`` is overloaded elsewhere as a generic "sequence
        is done" flag (set on length limit, repetition detection, and
        cross-rank completion sync — not just real EOS), so this method
        looks at the true cause of completion rather than just that bit.

        Side effects: emits a ``SeqEvent.COMPLETED`` lifespan event with
        the chosen reason; dumps the lifespan log on any non-stop finish
        or any context-mismatch observation.
        """
        # Repetition detected — dump lifespan for root cause analysis
        if seq._rep_detected:
            seq.log_event(SeqEvent.COMPLETED, ctx.rank, "finish_reason=repetition")
            lifespan.dump_lifespan(
                seq.uuid, seq.global_idx, seq._lifespan_log, "REPETITION_COMPLETE"
            )
            return "repetition"
        # Length truncation — per-sequence decode budget or model context limit
        if seq.decoded_length >= seq.max_decode_length:
            finish = "length"
        elif seq.current_context_length >= ctx.model_context_length:
            finish = "length"
        # Real EOS only — the token at seq.decoded_length-1 matches an EOS id
        elif seq.eos_reached and not ctx.ignore_eos:
            finish = "stop"
        else:
            finish = "length"
        # Log completion event
        seq.log_event(SeqEvent.COMPLETED, ctx.rank, f"finish_reason={finish}")
        # Dump lifespan if non-stop or any ctx mismatch was recorded
        if finish != "stop" or lifespan.has_ctx_mismatch(seq._lifespan_log):
            lifespan.dump_lifespan(
                seq.uuid, seq.global_idx, seq._lifespan_log, f"COMPLETE_{finish.upper()}"
            )
        return finish
