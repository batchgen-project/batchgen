"""Tests for `batchgen.cuda_graph.buckets.generate_bucket_sizes`."""

from __future__ import annotations

import pytest

from batchgen.cuda_graph.buckets import generate_bucket_sizes


def test_powers_of_two_pattern():
    # Plan §F A2: the canonical [256, 9] sweep must produce powers of two.
    assert generate_bucket_sizes(256, 9) == [1, 2, 4, 8, 16, 32, 64, 128, 256]


def test_dense_small_max():
    assert generate_bucket_sizes(16, 16) == list(range(1, 17))


def test_includes_endpoints():
    sizes = generate_bucket_sizes(128, 7)
    assert sizes[0] == 1
    assert sizes[-1] == 128


def test_sorted_unique():
    sizes = generate_bucket_sizes(256, 16)
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes)


def test_num_buckets_one():
    assert generate_bucket_sizes(64, 1) == [64]


def test_num_buckets_capped_at_max_bucket():
    # Cannot have more distinct buckets than max_bucket allows.
    sizes = generate_bucket_sizes(4, 16)
    assert sizes == [1, 2, 3, 4]


@pytest.mark.parametrize("max_bucket,num", [(256, 9), (256, 16), (128, 9), (16, 8)])
def test_returns_at_most_num_buckets(max_bucket, num):
    sizes = generate_bucket_sizes(max_bucket, num)
    assert len(sizes) <= num
    assert max(sizes) == max_bucket
    assert min(sizes) == 1
