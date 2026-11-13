"""Bucket-size generation for CUDA-graph capture.

Moved from `batchgen.batchgen_worker._generate_bucket_sizes` so adapters can
call it from `select_buckets` without depending on the worker module.
"""

from __future__ import annotations

import math
from typing import List


def generate_bucket_sizes(max_bucket: int, num_buckets: int) -> List[int]:
    """Generate exactly `num_buckets` bucket sizes from 1 to `max_bucket`.

    Uses geometric spacing for initial placement with magnitude-aware
    rounding (small values exact, large values rounded to clean multiples).
    Fills any gaps from rounding collisions by splitting the largest gaps.
    Caps at `max_bucket` if `num_buckets > max_bucket`.

    Examples:
      max=256, num=9  -> [1,2,4,8,16,32,64,128,256]
      max=256, num=16 -> [1,2,3,4,6,10,14,20,28,40,56,80,128,160,192,256]
      max=16,  num=16 -> [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    """
    num_buckets = min(num_buckets, max_bucket)
    if num_buckets <= 1:
        return [max_bucket]

    def _round_nice(x: float) -> int:
        if x <= 8:
            return int(round(x))
        log2 = int(math.log2(x))
        step = max(1 << (log2 - 2), 1)
        return max(1, round(x / step) * step)

    ratio = max_bucket ** (1.0 / (num_buckets - 1))
    sizes = set()
    for i in range(num_buckets):
        sizes.add(max(1, _round_nice(ratio ** i)))
    sizes.add(1)
    sizes.add(max_bucket)
    sizes_list = sorted(sizes)

    while len(sizes_list) < num_buckets:
        best_gap, best_idx = 0, -1
        for i in range(len(sizes_list) - 1):
            gap = sizes_list[i + 1] - sizes_list[i]
            if gap > best_gap:
                best_gap = gap
                best_idx = i
        if best_gap < 2:
            break
        mid = _round_nice((sizes_list[best_idx] + sizes_list[best_idx + 1]) / 2)
        if mid <= sizes_list[best_idx] or mid >= sizes_list[best_idx + 1]:
            mid = (sizes_list[best_idx] + sizes_list[best_idx + 1]) // 2
        if mid in sizes_list or mid <= sizes_list[best_idx] or mid >= sizes_list[best_idx + 1]:
            break
        sizes_list.insert(best_idx + 1, mid)

    return sizes_list


class DecodeWidthExceedsGraphBucketError(RuntimeError):
    """The per-rank decode width is wider than the largest captured bucket.

    On a deployment that requests whole-model decode graphs, decode admission
    is bounded by the per-rank decode width and the bucket ladder is derived
    to cover that width, so every admitted step has a graph to replay. A step
    or a plan wider than the ladder's top bucket would silently run eager; it
    is an invariant violation and fails loud instead.
    """


def decode_graph_bucket_sizes(
    max_bucket: int, num_buckets: int, decode_width: int
) -> List[int]:
    """Bucket ladder for whole-model decode graphs at a per-rank decode width.

    The ladder is ``generate_bucket_sizes`` over a top bucket of
    ``max(max_bucket, decode_width)``: a width above the configured top
    raises the top to the width, so every width decode admission can reach
    has a captured bucket. ``decode_width`` itself is always a bucket, so a
    step at full width replays its own graph instead of padding up to the
    next bucket.
    """
    max_bucket = int(max_bucket)
    decode_width = int(decode_width)
    if max_bucket <= 0:
        raise ValueError(f"max_bucket must be positive, got {max_bucket}")
    if decode_width <= 0:
        raise ValueError(f"decode_width must be positive, got {decode_width}")
    sizes = generate_bucket_sizes(max(max_bucket, decode_width), int(num_buckets))
    if decode_width not in sizes:
        sizes = sorted(set(sizes) | {decode_width})
    return sizes


def check_decode_width_within_buckets(
    decode_width: int, bucket_sizes: List[int]
) -> None:
    """Raise ``DecodeWidthExceedsGraphBucketError`` if no bucket covers the width."""
    top = max(int(b) for b in bucket_sizes) if bucket_sizes else 0
    if int(decode_width) > top:
        raise DecodeWidthExceedsGraphBucketError(
            f"per-rank decode width {int(decode_width)} exceeds the largest "
            f"whole-model decode graph bucket {top} (buckets="
            f"{sorted(int(b) for b in bucket_sizes)}); decode would run eager "
            "above the bucket. Raise --cuda-graph-max-bucket-size to at least "
            f"{int(decode_width)}."
        )


__all__ = [
    "DecodeWidthExceedsGraphBucketError",
    "check_decode_width_within_buckets",
    "decode_graph_bucket_sizes",
    "generate_bucket_sizes",
]
