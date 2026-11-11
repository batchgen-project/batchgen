"""Startup decode transient: device memory a decode step costs above steady state.

The GPU KV pool is sized from what is free once the decode model is resident;
the memory fraction's slack is what currently absorbs everything a decode
step or a graph capture allocates later. This module measures that transient
at the startup decode configure -- after ``configure_decoding`` at the
per-rank cap and after the pool is sized, before the worker reports ready --
so it can be published and, later, reserved by name.

Report-only: the measured value is logged and published with the deployment
capacity; nothing reads it for sizing.

Two quantities per execution flavor, both exact byte counts:

``allocator``
    Peak bytes the caching allocator handed out during the step, above what
    was allocated when it started (``max_memory_allocated`` after a peak
    reset, minus ``memory_allocated`` before).
``external``
    Growth of device memory the allocator does not own: the driver's used
    bytes minus the allocator's reserved bytes, after the step against before
    (communicator buffers, library workspaces, lazily loaded modules). The
    driver counter is instantaneous, so this is what the step leaves held,
    not an intra-step peak. Graph pools are allocator memory and land in
    ``reserved``, not here.

Flavors: ``eager`` runs a cap-sized synthetic decode step the model layer
provides (``startup_decode_probe_step(padding_bsz)`` on the parallel strategy
manager) twice, so the two runs can be checked against each other.
``capture`` wraps the decode graph capture the lifecycle already performs at
startup; a capture runs once per process, so it is measured once.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

logger = logging.getLogger(__name__)

GIB = 1024 ** 3

# The caching allocator rounds every large request up to a 2 MiB multiple, so
# two runs of one deterministic step may land at most one such block apart.
ALLOCATOR_BLOCK_BYTES = 2 * 1024 * 1024

FLAVOR_EAGER = "eager"
FLAVOR_CAPTURE = "capture"
FLAVORS = (FLAVOR_EAGER, FLAVOR_CAPTURE)

# Where the probe ran. Both are the startup decode configure; they differ only
# in which lifecycle reached it.
SEAM_PERSISTENT_BUILD = "startup-persistent-phase-build"
SEAM_DEPLOYMENT_PREPARE = "startup-deployment-prepare"


@dataclass(frozen=True)
class MemorySample:
    """Device memory counters at one instant, in bytes."""

    allocated: int
    peak_allocated: int
    reserved: int
    driver_used: int

    @property
    def external(self) -> int:
        """Driver-used bytes the caching allocator does not hold."""
        return self.driver_used - self.reserved


@dataclass(frozen=True)
class StartupTransient:
    """One flavor's measured transient on one rank, or why it is absent.

    ``deterministic`` is None when the flavor ran once (no second run to
    compare), and the two GiB fields are None exactly when ``unavailable``
    names a reason. GiB values are bytes / 2**30, exact in a float.
    """

    flavor: str
    seam: str
    padding_bsz: int
    allocator_gib: Optional[float] = None
    external_gib: Optional[float] = None
    deterministic: Optional[bool] = None
    unavailable: Optional[str] = None

    def __post_init__(self) -> None:
        if self.flavor not in FLAVORS:
            raise ValueError(f"unknown startup transient flavor {self.flavor!r}")
        measured = self.allocator_gib is not None and self.external_gib is not None
        if measured == (self.unavailable is not None):
            raise ValueError(
                "a startup transient is either measured (allocator_gib and "
                "external_gib) or unavailable with a reason, not both or neither"
            )

    @classmethod
    def missing(
        cls, flavor: str, seam: str, padding_bsz: int, reason: str
    ) -> "StartupTransient":
        return cls(flavor=flavor, seam=seam, padding_bsz=padding_bsz, unavailable=reason)

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def sample_device(cuda: Any, device: int) -> MemorySample:
    """Read the allocator and driver counters of ``device`` (no synchronize)."""
    free, total = cuda.mem_get_info(device)
    return MemorySample(
        allocated=int(cuda.memory_allocated(device)),
        peak_allocated=int(cuda.max_memory_allocated(device)),
        reserved=int(cuda.memory_reserved(device)),
        driver_used=int(total) - int(free),
    )


def _run_measured(cuda: Any, device: int, step: Callable[[], None]):
    """Run ``step`` once between synchronized samples; peak reset first."""
    cuda.synchronize(device)
    cuda.reset_peak_memory_stats(device)
    before = sample_device(cuda, device)
    step()
    cuda.synchronize(device)
    after = sample_device(cuda, device)
    return before, after


def allocator_transient_bytes(before: MemorySample, after: MemorySample) -> int:
    """Allocator peak during a run above what was allocated at its start."""
    return after.peak_allocated - before.allocated


def transient_runs_agree(
    first_bytes: int, second_bytes: int, block_bytes: int = ALLOCATOR_BLOCK_BYTES
) -> bool:
    """Determinism gate: two runs' allocator peaks within one allocator block.

    Only the allocator peak is compared. The external term of the first run
    includes one-time lazy creation (communicator buffers, workspaces) that
    the second run finds already present, so it differs between the runs by
    construction.
    """
    return abs(int(first_bytes) - int(second_bytes)) <= int(block_bytes)


def measure_eager(
    step: Callable[[], None], *, cuda: Any, device: int, seam: str, padding_bsz: int
) -> StartupTransient:
    """Run the synthetic decode step twice and gate the two allocator peaks.

    Published allocator value: the larger of the two peaks. External value:
    non-allocator growth across both runs together.

    Slice-1 deviation from the design: a mismatch is logged at ERROR and the
    struct records ``deterministic=False``; it does not fail startup, because
    nothing sizes from this value yet. Once the reservation feeds the pool
    budget, a mismatch must fail startup.
    """
    first_before, first_after = _run_measured(cuda, device, step)
    second_before, second_after = _run_measured(cuda, device, step)
    first = allocator_transient_bytes(first_before, first_after)
    second = allocator_transient_bytes(second_before, second_after)
    agree = transient_runs_agree(first, second)
    if not agree:
        logger.error(
            "[STARTUP_TRANSIENT] eager probe is not deterministic at seam=%s "
            "padding_bsz=%d: allocator peaks %d B and %d B differ by more than "
            "one allocator block (%d B); report-only, startup continues",
            seam, padding_bsz, first, second, ALLOCATOR_BLOCK_BYTES,
        )
    return StartupTransient(
        flavor=FLAVOR_EAGER,
        seam=seam,
        padding_bsz=padding_bsz,
        allocator_gib=max(first, second) / GIB,
        external_gib=(second_after.external - first_before.external) / GIB,
        deterministic=agree,
    )


def measure_capture(
    capture: Callable[[], None], *, cuda: Any, device: int, seam: str, padding_bsz: int
) -> StartupTransient:
    """Measure the lifecycle's startup graph capture; it runs exactly once."""
    before, after = _run_measured(cuda, device, capture)
    return StartupTransient(
        flavor=FLAVOR_CAPTURE,
        seam=seam,
        padding_bsz=padding_bsz,
        allocator_gib=allocator_transient_bytes(before, after) / GIB,
        external_gib=(after.external - before.external) / GIB,
    )


def reduce_across_ranks(
    per_rank: Sequence[Sequence[StartupTransient]],
) -> List[StartupTransient]:
    """Deployment value per flavor: the maximum over ranks.

    The tightest rank governs, as for the pool budget. A flavor absent on any
    rank is absent for the deployment (its maximum is unknown), with that
    rank's reason. Ranks must agree on the flavor order, seam and padding;
    a disagreement means they probed different configurations.
    """
    if not per_rank:
        raise ValueError("startup transient reduction needs at least one rank")
    reference = per_rank[0]
    reduced: List[StartupTransient] = []
    for index, head in enumerate(reference):
        column = []
        for rank, transients in enumerate(per_rank):
            if len(transients) != len(reference):
                raise ValueError(
                    f"rank {rank} reported {len(transients)} startup transient "
                    f"flavors, rank 0 reported {len(reference)}"
                )
            item = transients[index]
            if (item.flavor, item.seam, item.padding_bsz) != (
                head.flavor, head.seam, head.padding_bsz
            ):
                raise ValueError(
                    f"rank {rank} probed {item.flavor}/{item.seam}/"
                    f"padding_bsz={item.padding_bsz}, rank 0 probed "
                    f"{head.flavor}/{head.seam}/padding_bsz={head.padding_bsz}"
                )
            column.append((rank, item))
        absent = [(rank, item) for rank, item in column if item.unavailable is not None]
        if absent:
            rank, item = absent[0]
            reason = item.unavailable if rank == 0 else f"rank {rank}: {item.unavailable}"
            reduced.append(StartupTransient.missing(
                head.flavor, head.seam, head.padding_bsz, reason
            ))
            continue
        gates = [item.deterministic for _, item in column if item.deterministic is not None]
        reduced.append(StartupTransient(
            flavor=head.flavor,
            seam=head.seam,
            padding_bsz=head.padding_bsz,
            allocator_gib=max(item.allocator_gib for _, item in column),
            external_gib=max(item.external_gib for _, item in column),
            deterministic=all(gates) if gates else None,
        ))
    return reduced


def format_ledger_line(scope: str, transients: Sequence[StartupTransient]) -> str:
    """One memory-ledger line for a rank (``rank=N``) or the deployment."""
    parts = []
    for item in transients:
        if item.unavailable is not None:
            parts.append(f"{item.flavor}=unavailable({item.unavailable})")
        else:
            parts.append(
                f"{item.flavor}=allocator_gib:{item.allocator_gib:.6f},"
                f"external_gib:{item.external_gib:.6f},"
                f"deterministic:{item.deterministic}"
            )
    head = transients[0] if transients else None
    seam = head.seam if head is not None else "none"
    padding = head.padding_bsz if head is not None else -1
    return (
        f"[STARTUP_TRANSIENT] {scope} seam={seam} padding_bsz={padding} "
        f"reserved_for_sizing=no {' '.join(parts)}"
    )


def log_ledger_line(scope: str, transients: Sequence[StartupTransient]) -> None:
    logger.info("%s", format_ledger_line(scope, transients))
