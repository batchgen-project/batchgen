from batchgen.sequence import SequenceEntry

CONTEXT = 131072


def _seq(max_tokens):
    return SequenceEntry(uuid="s", global_idx=0, prompt_length=0, max_decode_length=max_tokens)


def test_overflowing_request_is_clamped_consistently():
    seq = _seq(131072)
    budget = seq.clamp_decode_to_context(1100, CONTEXT)
    seq.original_prompt_length = 1100
    assert budget == CONTEXT
    assert seq.max_decode_length == seq.original_max_decode_length == CONTEXT - 1100
    # The invariant checked by validate_metadata on every rank.
    assert budget == seq.original_prompt_length + seq.original_max_decode_length


def test_request_within_context_is_unchanged():
    seq = _seq(4096)
    assert seq.clamp_decode_to_context(1100, CONTEXT) == 1100 + 4096
    assert seq.max_decode_length == seq.original_max_decode_length == 4096
