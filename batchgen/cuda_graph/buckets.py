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


__all__ = ["generate_bucket_sizes"]
