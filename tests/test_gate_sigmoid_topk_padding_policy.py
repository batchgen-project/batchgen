"""CPU contract for ``gate_sigmoid_topk_cuda``'s padding-shape arguments.

The kernel accepts two mutually exclusive descriptions of which rows are live:

  * ``num_valid_tokens`` — one device scalar; the live rows are a prefix of N.
  * ``rank_token_counts`` + ``rows_per_rank`` — the EP all-gathered layout, R
    contiguous blocks of ``rows_per_rank`` rows each live up to its own count,
    so the live rows are NOT a prefix of N.

Combining them, or giving half of the per-rank pair, is a caller bug. These
cases must raise a named Python error BEFORE the CUDA extension is loaded, so
they are reachable on a machine with no GPU and they cannot surface as a
CUDA-side abort in the middle of a graph capture.
"""

import pytest

from batchgen.moe.routing import gate_sigmoid_topk_cuda


_SENTINEL = object()  # never dereferenced: every case must fail before the ext loads


def _call(**kwargs):
    return gate_sigmoid_topk_cuda(
        _SENTINEL,  # router_logits
        _SENTINEL,  # e_score_correction
        **kwargs,
    )


def test_rank_counts_and_num_valid_tokens_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _call(num_valid_tokens=_SENTINEL, rank_token_counts=_SENTINEL, rows_per_rank=8)


def test_rank_counts_without_rows_per_rank_fails():
    with pytest.raises(ValueError, match="rows_per_rank is required"):
        _call(rank_token_counts=_SENTINEL)


@pytest.mark.parametrize("rows_per_rank", [0, -1])
def test_non_positive_rows_per_rank_fails(rows_per_rank):
    with pytest.raises(ValueError, match="rows_per_rank must be positive"):
        _call(rank_token_counts=_SENTINEL, rows_per_rank=rows_per_rank)


def test_rows_per_rank_without_rank_counts_fails():
    with pytest.raises(ValueError, match="only meaningful together"):
        _call(rows_per_rank=8)


def test_latent_out_without_offset_fails():
    with pytest.raises(ValueError, match="latent_offset is required"):
        _call(latent_out=_SENTINEL)
