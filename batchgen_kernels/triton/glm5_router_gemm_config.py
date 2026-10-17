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

# Split count for the dispatched row range, measured rather than assumed, and
# deliberately ONE count for the whole range.
#
# Splitting turns one program per output tile into ``split_k`` programs, each
# walking ``k // split_k`` of the K axis. At these row counts that lifts the
# program count from well under the device's multiprocessor count to several
# programs per multiprocessor, which is what the serial-K shape cannot do: it
# has no second program to hide its K-loop memory latency behind. Past the
# point where the device is full, more splits only shorten each K walk while
# growing the partial buffer and the reduction pass.
#
# The entry is the count a microbenchmark found fastest, taken as the minimum
# over every exact split count from 1 to 32 rather than from a two-way
# comparison, with the count forced per run through ``split_k=`` and the kernel
# timed under graph replay at the production K and expert count.
#
# The measurement has to hold the ROUTER WEIGHT the way production holds it:
# one distinct weight per routed layer, cycled through the timed region so that
# no call finds its weight already resident in L2. An earlier sweep reused a
# single weight across every call it timed, which left that one weight
# L2-resident and flattered a latency-bound, low-occupancy launch well beyond
# how it runs in the model -- far enough to move which count is fastest at the
# bottom of the range and to understate the unsplit kernel outright. The
# cold-weight bench agrees with an in-situ profiler trace of the real server to
# a fraction of a percent and the warm one does not, so the cold-weight numbers
# are the record here and the warm sweep is superseded. Both harnesses, the
# sweep and the calibration against that trace: internal campaign
# glm52-decode-step-gap-vs-sglang, slice F, which is the record to re-run
# before changing the entry.
#
# One count covers the range by ruling rather than by coincidence. The count
# fixes the fp32 reduction order, so a compact launch reducing at one count
# while the same rows inside a padded larger bucket reduce at another makes a
# valid row's logits differ between the two launches -- which is exactly the
# cross-bucket bitwise M-independence contract the suites assert. Where the
# per-row fastest count differs from the one stated here it does so only at the
# very top of the range and by a small margin, against a recovery at the bottom
# of the range that is larger by an order of magnitude; an earlier revision of
# this file carried a per-bucket table and was withdrawn for the same reason.
# Ruled by the campaign above, 2026-10-17: the bitwise invariant wins, so the
# count is constant and ``_validate_measured_table`` enforces that constancy at
# import rather than trusting this literal.
ROUTER_SPLIT_K_MEASURED: tuple[tuple[int, int], ...] = ((64, 8),)

# The distinct split counts the table can select -- one, by the constancy the
# validation enforces. It has to partition the production K exactly, and it is
# covered by the exactness suites.
ROUTER_SPLIT_K_VALUES: tuple[int, ...] = tuple(
    sorted({split for _, split in ROUTER_SPLIT_K_MEASURED})
)

# Row range served by the split-K path, and the only place those bounds are
# stated: the GLM-5 MoE graph segment dispatches its router to this kernel for
# exactly this range, so it reads the bounds from here rather than repeating
# them.
#
# The lower bound is the smallest row count the graphed router is dispatched
# at. It is where the serial-K shape is at its worst -- fewest programs, so the
# least able to hide its K-loop memory latency -- and the cold-weight
# measurement finds the split-K pair ahead of it by the widest margin of the
# whole range there, so there is no row count in the dispatched range the
# serial-K kernel should be preferred at. Above the upper bound the graphed
# router path uses a different kernel entirely. The measured table has to be
# anchored at the lower bound and may not claim a measurement above the upper
# one -- ``_validate_measured_table`` below enforces both, so the range and the
# table cannot drift apart. It needs no upper anchor: one count covers the
# range by construction.
ROUTER_SPLIT_K_MIN_ROWS = 64
ROUTER_SPLIT_K_MAX_ROWS = 512


class RouterGemmShapeError(ValueError):
    """A router GEMM shape the requested split count cannot serve exactly."""


class RouterGemmPolicyError(ValueError):
    """The measured split-count table does not describe the captured range."""


def _validate_measured_table() -> None:
    """Fail at import unless the table states one split for the dispatched range.

    The table is the only place the split count comes from, and the count is
    what fixes the fp32 reduction order, so two things have to hold before
    anything imports this module. The table has to be anchored in the row range
    the GLM-5 MoE graph segment hands this kernel, rather than describing rows
    that never reach the split-K path; and it has to state a SINGLE count for
    that range. The second is the load-bearing one: a table whose entries
    disagree would give the same row a different reduction order in a compact
    launch and in a padded bucket, which is a bitwise difference the
    cross-bucket M-independence contract forbids. Checked here instead of
    trusting the literal, loudly and once.
    """
    if not ROUTER_SPLIT_K_MEASURED:
        raise RouterGemmPolicyError("router split-K table is empty")
    rows = [row for row, _ in ROUTER_SPLIT_K_MEASURED]
    if rows != sorted(set(rows)):
        raise RouterGemmPolicyError(
            f"router split-K table rows must be strictly increasing, got {rows}"
        )
    if rows[0] != ROUTER_SPLIT_K_MIN_ROWS:
        raise RouterGemmPolicyError(
            f"router split-K table starts at row {rows[0]}, not at the bottom "
            f"{ROUTER_SPLIT_K_MIN_ROWS} of the dispatched range "
            f"{ROUTER_SPLIT_K_MIN_ROWS}..{ROUTER_SPLIT_K_MAX_ROWS}"
        )
    if rows[-1] > ROUTER_SPLIT_K_MAX_ROWS:
        raise RouterGemmPolicyError(
            f"router split-K table measures row {rows[-1]}, above the "
            f"dispatched range {ROUTER_SPLIT_K_MIN_ROWS}.."
            f"{ROUTER_SPLIT_K_MAX_ROWS}"
        )
    splits = sorted({split for _, split in ROUTER_SPLIT_K_MEASURED})
    if len(splits) != 1:
        raise RouterGemmPolicyError(
            f"router split-K table states counts {splits} across the "
            f"dispatched range {ROUTER_SPLIT_K_MIN_ROWS}.."
            f"{ROUTER_SPLIT_K_MAX_ROWS}; the split count fixes the float32 "
            "reduction order, so one count has to cover the whole range or a "
            "valid row's logits stop being bitwise independent of how many "
            "padded rows shared its launch"
        )
    if splits[0] < 2:
        raise RouterGemmPolicyError(
            f"router split-K table entry split_k={splits[0]} is not a split"
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
    """The measured split count for ``m`` rows -- the same count for every ``m``.

    Not every captured bucket was measured, and none has to be:
    ``generate_bucket_sizes`` can place a bucket anywhere, so
    ``world_size * bucket_size`` reaches row counts such as 200 or 320 that no
    sweep point sits on, and they all take the one count
    :data:`ROUTER_SPLIT_K_MEASURED` states. ``m`` stays in the signature
    because it states what the count is allowed to depend on -- the row count
    and K, never how many rows of a launch are live -- and keeps this the one
    seam a future measurement would reopen; it does not vary the answer, which
    is what ``_validate_measured_table`` enforces.

    An earlier revision resolved ``m`` against a per-bucket table by nearest
    measured point. See the note on :data:`ROUTER_SPLIT_K_MEASURED` for why a
    bucket-dependent count was withdrawn.
    """
    return ROUTER_SPLIT_K_MEASURED[0][1]


def select_router_split_k(m: int, k: int) -> int:
    """Return the split count for an ``m x k`` router GEMM.

    Returns 1 -- the serial-K kernel -- for every shape outside the row range
    the split-K path targets, and for any K the measured count for those rows
    cannot partition exactly.

    The count is constant over the whole captured row range, so every bucket's
    graph bakes in the same split count, the same K partition and the same
    reduction order, and a row's logits are bitwise the same whether its launch
    carried the padded bucket or only the valid rows. Keeping it a function of
    ``m`` and ``k`` alone is what makes that true at capture time -- ``m`` is
    ``world_size * bucket_size``, static per captured bucket -- and what the
    split-K path forbids outright is a count derived from how many rows of a
    launch are live, which would make a row's result depend on its padding.
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
