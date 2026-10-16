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

# Split count for the router shapes that take the split-K path.
#
# Fixed, not derived from the launch: the K partition -- and therefore the
# summation order of every output element -- has to be identical on every
# launch and for every captured bucket size. Deriving the split from M would
# make a row's result depend on how much padding shared its launch, which the
# graphed router path forbids.
#
# Eight splits turn one program per output tile into eight, each walking an
# eighth of K. At the row counts below that lifts the program count from well
# under the device's multiprocessor count to several programs per
# multiprocessor, which is what the serial-K shape cannot do: it has no second
# program to hide its K-loop memory latency behind. Raising it further shortens
# each K walk but grows the partial buffer and the reduction pass, so the
# useful range is narrow; a microbenchmark picks the final value by passing
# ``split_k=`` explicitly to the kernel.
ROUTER_SPLIT_K = 8

# Row range served by the split-K path. The lower bound is where the serial-K
# program count stops being large enough to fill the device; above the upper
# bound the graphed router path uses a different kernel entirely, so these
# bounds mirror the dispatch the GLM-5 MoE graph segment already performs.
ROUTER_SPLIT_K_MIN_ROWS = 192
ROUTER_SPLIT_K_MAX_ROWS = 512


class RouterGemmShapeError(ValueError):
    """A router GEMM shape the requested split count cannot serve exactly."""


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


def select_router_split_k(m: int, k: int) -> int:
    """Return the split count for an ``m x k`` router GEMM.

    Returns 1 -- the serial-K kernel -- for every shape outside the row range
    the split-K path targets, and for any K that cannot be partitioned exactly.
    """
    if m < ROUTER_SPLIT_K_MIN_ROWS or m > ROUTER_SPLIT_K_MAX_ROWS:
        return 1
    if not router_split_k_is_exact(k, ROUTER_SPLIT_K):
        return 1
    return ROUTER_SPLIT_K


def router_split_k_bounds(k: int, split_k: int) -> tuple[tuple[int, int], ...]:
    """The ``[start, stop)`` K range each split accumulates, in reduction order.

    Depends only on ``k`` and ``split_k``. The reduction then sums the splits
    in this same order, index 0 upwards, so an output element's float32
    operation order is fully determined by the K axis and never by ``m``.
    """
    validate_router_split_k(k, split_k)
    per = k // split_k
    return tuple((s * per, (s + 1) * per) for s in range(split_k))
