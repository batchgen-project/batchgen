"""Tests for the bucket ladders in `batchgen.cuda_graph.buckets`."""

from __future__ import annotations

import pytest

from batchgen.cuda_graph.buckets import (
    DecodeWidthExceedsGraphBucketError,
    check_decode_width_within_buckets,
    decode_graph_bucket_sizes,
    generate_bucket_sizes,
)


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


# ---- decode graph ladder at a per-rank decode width ------------------------


def test_decode_ladder_equals_configured_ladder_when_width_is_its_top():
    # The default deployment: width 128 == --cuda-graph-max-bucket-size 128.
    assert decode_graph_bucket_sizes(128, 16, 128) == generate_bucket_sizes(128, 16)


def test_decode_ladder_top_is_raised_to_a_wider_decode_width():
    sizes = decode_graph_bucket_sizes(128, 16, 160)
    assert max(sizes) == 160
    assert 160 in sizes
    assert sizes == generate_bucket_sizes(160, 16)


def test_decode_ladder_keeps_a_wider_configured_top_and_adds_the_width():
    # 200-bucket ladder has no 160 bucket; a full-width step must not pad
    # up to 192.
    assert 160 not in generate_bucket_sizes(200, 16)
    sizes = decode_graph_bucket_sizes(200, 16, 160)
    assert max(sizes) == 200
    assert 160 in sizes
    assert sizes == sorted(set(sizes))
    assert set(generate_bucket_sizes(200, 16)) <= set(sizes)


@pytest.mark.parametrize("flag,num,width", [
    (128, 16, 128), (128, 16, 160), (64, 9, 128), (256, 16, 128),
    (200, 16, 160), (8, 4, 3), (1, 1, 1), (16, 1, 32),
])
def test_decode_ladder_always_covers_and_contains_the_width(flag, num, width):
    sizes = decode_graph_bucket_sizes(flag, num, width)
    assert width in sizes
    assert max(sizes) == max(flag, width)
    assert sizes == sorted(set(sizes))
    check_decode_width_within_buckets(width, sizes)


@pytest.mark.parametrize("flag,width", [(0, 128), (128, 0), (-1, 8)])
def test_decode_ladder_rejects_non_positive_sizes(flag, width):
    with pytest.raises(ValueError, match="must be positive"):
        decode_graph_bucket_sizes(flag, 16, width)


def test_width_check_names_the_flag_when_the_width_exceeds_the_top_bucket():
    with pytest.raises(DecodeWidthExceedsGraphBucketError) as info:
        check_decode_width_within_buckets(160, generate_bucket_sizes(128, 16))
    message = str(info.value)
    assert "per-rank decode width 160" in message
    assert "largest whole-model decode graph bucket 128" in message
    assert "Raise --cuda-graph-max-bucket-size to at least 160" in message


def test_width_check_accepts_the_top_bucket_exactly_and_rejects_an_empty_ladder():
    check_decode_width_within_buckets(128, [1, 64, 128])
    with pytest.raises(DecodeWidthExceedsGraphBucketError):
        check_decode_width_within_buckets(1, [])


def test_width_error_is_a_runtime_error():
    assert issubclass(DecodeWidthExceedsGraphBucketError, RuntimeError)
