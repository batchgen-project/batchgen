"""Launch policy for the GLM-5 router GEMM.

Deliberately free of Triton and CUDA imports: the split-K selection rule and
the K partition it implies are the parts that carry the kernel's correctness
contract, so they stay importable — and testable — without a GPU toolchain.
"""

from __future__ import annotations

# M/N/K tiling shared by the serial-K kernel and the split-K partial kernel.
# Split-K must not change the tile shape. With the same block_k, one split's
# partial sum covers exactly the same per-block_k terms as the serial kernel
# would, in the same order, so the two results differ only in how those terms
# are grouped -- a pure float32 reassociation, bounded by the accumulator
# precision rather than by anything shape dependent.
ROUTER_BLOCK_M = 16
ROUTER_BLOCK_N = 64
ROUTER_BLOCK_K = 64

# Split count per captured bucket, measured rather than assumed.
#
# Splitting turns one program per output tile into ``split_k`` programs, each
# walking ``k // split_k`` of the K axis. At these row counts that lifts the
# program count from well under the device's multiprocessor count to several
# programs per multiprocessor, which is what the serial-K shape cannot do: it
# has no second program to hide its K-loop memory latency behind. Past the
# point where the device is full, more splits only shorten each K walk while
# growing the partial buffer and the reduction pass, so the best count falls as
# the row count rises -- which is exactly what the sweep below shows.
#
# Each entry is the count a microbenchmark found fastest for that bucket, with
# the count forced per run through ``split_k=`` and the kernel timed under graph
# replay at the production K and expert count. Sweep and figures: internal
# campaign glm52-decode-step-gap-vs-sglang, slice A, which is the record to
# re-run before changing an entry. The shape of the result is that only the
# lowest captured bucket is still short of programs; from 256 rows up the extra
# reduction pass costs more than the shorter K walk saves, and the margin
# against the higher count widens with the row count.
ROUTER_SPLIT_K_MEASURED: tuple[tuple[int, int], ...] = (
    (192, 8),
    (256, 4),
    (384, 4),
    (512, 4),
)

# The distinct split counts the table can select. Both have to partition the
# production K exactly, and both are covered by the exactness suites.
ROUTER_SPLIT_K_VALUES: tuple[int, ...] = tuple(
    sorted({split for _, split in ROUTER_SPLIT_K_MEASURED})
)

# Row range served by the split-K path. The lower bound is where the serial-K
# program count stops being large enough to fill the device; above the upper
# bound the graphed router path uses a different kernel entirely, so these
# bounds mirror the dispatch the GLM-5 MoE graph segment already performs. The
# measured table has to span exactly this range -- ``_validate_measured_table``
# below enforces it, so the range and the table cannot drift apart.
ROUTER_SPLIT_K_MIN_ROWS = 192
ROUTER_SPLIT_K_MAX_ROWS = 512


class RouterGemmShapeError(ValueError):
    """A router GEMM shape the requested split count cannot serve exactly."""


class RouterGemmPolicyError(ValueError):
    """The measured split-count table does not describe the captured range."""


def _validate_measured_table() -> None:
    """Fail at import if the table stops covering the dispatched row range.

    The table is the only place the split count comes from, so a table that
    does not span the rows the GLM-5 MoE graph segment hands this kernel would
    silently resolve some bucket by extrapolation. Checked here instead, loudly
    and once.
    """
    if not ROUTER_SPLIT_K_MEASURED:
        raise RouterGemmPolicyError("router split-K table is empty")
    rows = [row for row, _ in ROUTER_SPLIT_K_MEASURED]
    if rows != sorted(set(rows)):
        raise RouterGemmPolicyError(
            f"router split-K table rows must be strictly increasing, got {rows}"
        )
    if rows[0] != ROUTER_SPLIT_K_MIN_ROWS or rows[-1] != ROUTER_SPLIT_K_MAX_ROWS:
        raise RouterGemmPolicyError(
            f"router split-K table spans rows {rows[0]}..{rows[-1]}, which does "
            f"not match the dispatched range "
            f"{ROUTER_SPLIT_K_MIN_ROWS}..{ROUTER_SPLIT_K_MAX_ROWS}"
        )
    for _, split in ROUTER_SPLIT_K_MEASURED:
        if split < 2:
            raise RouterGemmPolicyError(
                f"router split-K table entry split_k={split} is not a split"
            )


_validate_measured_table()


def router_split_k_is_exact(k: int, split_k: int) -> bool:
    """Whether ``split_k`` partitions ``k`` into equal whole-block_k chunks.

    An exact partition is what lets the partial kernel run a compile-time
    number of block_k steps with no K masking, and it is also what makes the
    chunk boundaries a function of ``k`` alone.
    """
    if split_k < 1:
        return False
    if split_k == 1:
        return True
    if k % split_k:
        return False
    return (k // split_k) % ROUTER_BLOCK_K == 0


def validate_router_split_k(k: int, split_k: int) -> None:
    """Raise :class:`RouterGemmShapeError` unless the partition is exact."""
    if not router_split_k_is_exact(k, split_k):
        raise RouterGemmShapeError(
            f"router GEMM split_k={split_k} cannot partition k={k} into equal "
            f"chunks of a whole multiple of block_k={ROUTER_BLOCK_K}; "
            "split_k must divide k and k // split_k must divide by block_k"
        )


def router_measured_split_k(m: int) -> int:
    """The measured-best split count for ``m`` rows, by nearest measured point.

    Not every captured bucket was measured: ``generate_bucket_sizes`` can place
    a bucket anywhere, so ``world_size * bucket_size`` reaches row counts such
    as 224 or 320 that sit between the points in
    :data:`ROUTER_SPLIT_K_MEASURED`. Those take the nearest measured point's
    count rather than an interpolation, because the quantity being chosen is a
    discrete split count and the table records where it changes, not a curve.

    Ties -- 224 is 32 rows from both 192 and 256 -- resolve to the higher row
    count. The higher split count's advantage is confined to the bottom of the
    captured range and has already inverted by 256 rows, while the penalty for
    a count that is too high widens as the row count grows, so the tie goes to
    the count that holds over the rest of the range.
    """
    best_rows, best_split = ROUTER_SPLIT_K_MEASURED[0]
    for rows, split in ROUTER_SPLIT_K_MEASURED[1:]:
        if abs(m - rows) <= abs(m - best_rows):
            best_rows, best_split = rows, split
    return best_split


def select_router_split_k(m: int, k: int) -> int:
    """Return the split count for an ``m x k`` router GEMM.

    Returns 1 -- the serial-K kernel -- for every shape outside the row range
    the split-K path targets, and for any K the measured count for those rows
    cannot partition exactly.

    The count varies ACROSS captured buckets and is constant WITHIN one: ``m``
    is ``world_size * bucket_size``, static per captured bucket, so the graphed
    router path resolves this at capture time and bakes one split count -- and
    with it one K partition and one reduction order -- into each bucket's graph.
    What the split-K path still forbids is a count derived from how many rows of
    a launch are live, which would make a row's result depend on its padding.
    """
    if m < ROUTER_SPLIT_K_MIN_ROWS or m > ROUTER_SPLIT_K_MAX_ROWS:
        return 1
    split_k = router_measured_split_k(m)
    if not router_split_k_is_exact(k, split_k):
        return 1
    return split_k


def router_split_k_bounds(k: int, split_k: int) -> tuple[tuple[int, int], ...]:
    """The ``[start, stop)`` K range each split accumulates, in reduction order.

    Depends only on ``k`` and ``split_k``. The reduction then sums the splits
    in this same order, index 0 upwards, so an output element's float32
    operation order is fully determined by the K axis and never by ``m``.
    """
    validate_router_split_k(k, split_k)
    per = k // split_k
    return tuple((s * per, (s + 1) * per) for s in range(split_k))
