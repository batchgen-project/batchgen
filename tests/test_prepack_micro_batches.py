import pytest

from batchgen.prefill.prepack import build_prefill_micro_batches


def test_build_prefill_micro_batches_obeys_token_cap_without_l2_balance():
    micro_batches, l2_cap = build_prefill_micro_batches(
        [10, 20, 30, 40],
        token_cap=50,
        l2_balance=False,
    )

    assert micro_batches == [(0, 2), (2, 3), (3, 4)]
    assert l2_cap == 0


def test_build_prefill_micro_batches_can_force_single_sequence_batches():
    micro_batches, l2_cap = build_prefill_micro_batches(
        [1138, 1439],
        token_cap=4096,
        single_sequence_only=True,
    )

    assert micro_batches == [(0, 1), (1, 2)]
    assert l2_cap == 0


def test_build_prefill_micro_batches_obeys_page_cap():
    micro_batches, l2_cap = build_prefill_micro_batches(
        [100, 100, 100, 100],
        token_cap=1000,
        page_lengths=[3, 4, 5, 6],
        page_cap=8,
        l2_balance=False,
    )

    assert micro_batches == [(0, 2), (2, 3), (3, 4)]
    assert l2_cap == 0


def test_build_prefill_micro_batches_keeps_oversized_sequence_whole():
    micro_batches, l2_cap = build_prefill_micro_batches(
        [100, 100],
        token_cap=1000,
        page_lengths=[12, 1],
        page_cap=8,
        l2_balance=False,
    )

    assert micro_batches == [(0, 1), (1, 2)]
    assert l2_cap == 0


def test_build_prefill_micro_batches_requires_positive_token_cap():
    with pytest.raises(ValueError, match="token_cap must be positive"):
        build_prefill_micro_batches([16, 32], token_cap=0)
