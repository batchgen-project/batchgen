from batchgen.sequence import SequenceBatch, SequenceEntry, SequenceStatus


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

    batch.update_status("seq-c", SequenceStatus.IN_PREFILL)

    assert batch.get_sequences_for_rank_with_status(0, SequenceStatus.QUEUEING) == [
        "seq-a",
        "seq-b",
    ]
