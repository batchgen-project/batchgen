"""Phase 2.8.2e — port of ``_decode_update_sequences`` (8377-8448).

Per-step per-sequence bookkeeping after a forward pass:

  * Skip completed sequences.
  * Record the sampled token into the query-book buffer pool via
    ``adapter.record_decoded_token``.
  * Advance ``decoded_length`` + ``current_context_length``.
  * Detect EOS (``adapter.should_stop_at_eos``) and max-decode length.
  * Consecutive-token + variable-length N-gram repetition detection.

All state mutations on :class:`SequenceEntry` happen here (native);
buffer writes + N-gram pattern reads route through the adapter.
"""

from __future__ import annotations

import os
from typing import Any

from batchgen.lifespan import SeqEvent
from batchgen.sequence import SequenceEntry
from batchgen.worker.protocols import UUID, LegacyInfraBackend
from batchgen.worker.state import WorkerState


_REP_DETECTION = os.environ.get("BATCHGEN_REP_DETECTION", "1") == "1"


def update_sequences(
    state: WorkerState,
    adapter: LegacyInfraBackend,
    *,
    batch: list[int],
    new_tokens_cpu: Any,        # torch.Tensor, CPU
    local_iteration: int,
) -> None:
    """Apply one forward pass's sampled tokens to the batch sequences.

    Parameters:
        batch: rank-local indices aligned with the forward output.
        new_tokens_cpu: sampled tokens already moved to CPU (shape
            ``(batch, 1)`` or ``(batch,)`` — both supported).
        local_iteration: monotonic iteration counter for the current
            ``run_continuous`` call. Not used by update logic itself;
            kept for parity with legacy (which may later surface
            diagnostics).
    """
    local_to_uuid = adapter.local_to_uuid_map()
    for i, local_idx in enumerate(batch):
        uuid = local_to_uuid.get(local_idx)
        if uuid is None:
            continue
        seq = state.global_batch.get_sequence(uuid)
        if seq is None:
            continue
        if adapter.is_sequence_completed(seq):
            continue

        decode_pos = seq.decoded_length
        token_tensor = new_tokens_cpu[i]
        adapter.record_decoded_token(
            local_idx=local_idx,
            decode_pos=decode_pos,
            token=token_tensor,
        )

        seq.decoded_length += 1
        seq.current_context_length += 1

        token_id = int(token_tensor.item())

        if adapter.should_stop_at_eos(token_id):
            seq.eos_reached = True

        if seq.decoded_length >= seq.max_decode_length:
            seq.eos_reached = True

        if _REP_DETECTION and not seq._rep_detected:
            _detect_repetition(
                state, adapter, seq, local_idx, token_id,
            )


def _detect_repetition(
    state: WorkerState,
    adapter: LegacyInfraBackend,
    seq: SequenceEntry,
    local_idx: int,
    token_id: int,
) -> None:
    """Consecutive-token + N-gram repetition check.

    Mirrors legacy 8420-8447. Runs only when ``BATCHGEN_REP_DETECTION``
    env var enables it and the sequence has not already been flagged.
    """
    # Consecutive same-token counter.
    if token_id == seq._rep_last_token:
        seq._rep_count += 1
        if seq._rep_count >= 32:
            seq._rep_detected = True
            seq.eos_reached = True
            seq.log_event(
                SeqEvent.REPETITION, state.rank,
                f"token={token_id}, count={seq._rep_count}",
            )
            return
    else:
        seq._rep_last_token = token_id
        seq._rep_count = 1

    # Variable-length N-gram pattern check every 64 decoded tokens.
    if (
        not seq._rep_detected
        and seq.decoded_length >= 6
        and seq.decoded_length % 64 == 0
    ):
        if adapter.check_repeating_ngram_pattern(
            local_idx=local_idx, decoded_length=seq.decoded_length,
        ):
            seq._rep_detected = True
            seq.eos_reached = True
            seq.log_event(
                SeqEvent.REPETITION, state.rank,
                f"ngram at decoded_length={seq.decoded_length}",
            )


__all__ = ["update_sequences"]
