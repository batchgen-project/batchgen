from batchgen.sequence import SequenceBatch, SequenceEntry, SequenceStatus
from batchgen.worker.admission.plan import StatusEdge, Transition, TransitionPlan, apply_or_raise

_EDGE_FOR = {
    SequenceStatus.IN_PREFILL: StatusEdge.PREFILL,
    SequenceStatus.PREFILLED: StatusEdge.PREFILL,
    SequenceStatus.IN_DECODE: StatusEdge.DECODE_ENTRY,
    SequenceStatus.ON_HOLD: StatusEdge.BOUNDARY,
    SequenceStatus.EVICTED: StatusEdge.BOUNDARY,
    SequenceStatus.COMPLETED: StatusEdge.COMPLETION,
}


def _move(batch, uuid, status):
    """Move one sequence through ``SequenceBatch.apply``; a rejection raises."""
    apply_or_raise(batch, TransitionPlan(_EDGE_FOR[status], (Transition(uuid, status),)))


def test_get_sequences_for_rank_returns_global_idx_order():
    batch = SequenceBatch()
    seqs = [
        SequenceEntry("seq-c", global_idx=20, prompt_length=8, max_decode_length=16),
        SequenceEntry("seq-a", global_idx=3, prompt_length=8, max_decode_length=16),
        SequenceEntry("seq-b", global_idx=11, prompt_length=8, max_decode_length=16),
    ]
    for seq in seqs:
        batch.add_sequence(seq)
        batch.assign_rank(seq.uuid, 0)

    assert batch.get_sequences_for_rank(0) == ["seq-a", "seq-b", "seq-c"]


def test_get_sequences_for_rank_with_status_returns_global_idx_order():
    batch = SequenceBatch()
    seqs = [
        SequenceEntry("seq-c", global_idx=20, prompt_length=8, max_decode_length=16),
        SequenceEntry("seq-a", global_idx=3, prompt_length=8, max_decode_length=16),
        SequenceEntry("seq-b", global_idx=11, prompt_length=8, max_decode_length=16),
    ]
    for seq in seqs:
        batch.add_sequence(seq)
        batch.assign_rank(seq.uuid, 0)

    _move(batch, "seq-c", SequenceStatus.IN_PREFILL)

    assert batch.get_sequences_for_rank_with_status(0, SequenceStatus.QUEUEING) == [
        "seq-a",
        "seq-b",
    ]
