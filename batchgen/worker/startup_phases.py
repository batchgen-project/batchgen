"""Named startup phases, shared lock-free between the server and its workers.

A cold start is a sequence of long, silent steps: the weight-store copy, the
runtime-extension preloads, the communicator rendezvous, the decode shard
build. A step that never returns -- a network-filesystem read that stalls
inside an uninterruptible syscall, a rendezvous a peer never joins -- looks
from the outside exactly like a step that is merely slow: every rank keeps
its GPU busy, nothing is logged, and the launch neither fails nor finishes.

This module is the seam that makes that state readable and bounded. Every
process that runs a startup phase stores the phase's name and the time it
entered it into its own slot of one shared record; the server's watchdog
reads every slot and, when a slot has sat in one phase past that phase's
budget, names the slot, the phase and the seconds stuck in the log BEFORE it
fails the startup. The budget is per phase ENTRY, so a startup that is slow
but moving from phase to phase is never killed.

The record is deliberately a lock-free shared array, for the same reason
``WorkerManager._cohort_shutdown_flag`` is: the reader must be able to read
the slot of a rank that is stuck below Python or already dead, so neither
side may take a lock the other could be holding. A per-slot sequence counter
(odd while a write is in flight) lets the reader reject a torn pair instead
of mistaking a half-written slot for a stall. Exactly one process writes any
given slot, so the counter needs no atomic increment.

``time.monotonic`` is the clock on both sides. On Linux it is
``CLOCK_MONOTONIC``, which is system-wide rather than per process, and a
worker always runs on the host of the manager that spawned it, so an entry
time written by a worker is directly comparable in the manager.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# Phase names. They are generic startup phases, not one model's steps: the
# same names are reported by whatever model's startup path runs them.
#: The server's own copy of the weight store into its staging tmpfs.
PHASE_STORE_STAGING = "store-staging"
#: Any rendezvous that must complete before compute can start: the process
#: group, the NCCL warmup collectives, the side communicator.
PHASE_COMMUNICATOR_INIT = "communicator-init"
#: Building the engine and the model from the checkpoint.
PHASE_MODEL_BUILD = "model-build"
#: Loading the family's required CUDA extensions.
PHASE_EXTENSION_PRELOAD = "extension-preload"
#: Loading the selected model package's own extensions on top of them.
PHASE_MODEL_EXTENSION_PRELOAD = "model-extension-preload"
#: Materializing the resident decode expert shards.
PHASE_DECODE_SHARD_BUILD = "decode-shard-build"
#: Configuring the prefill phase and its weight-copy pipeline.
PHASE_PREFILL_CONFIGURE = "prefill-configure"
#: One-time buffer/handshake prewarm that must not land on the first request.
PHASE_PREWARM = "prewarm"

#: Every phase a startup may report, in wire order: a record stores a phase's
#: INDEX in this tuple, so a name may be appended but never moved or removed.
STARTUP_PHASES: Tuple[str, ...] = (
    PHASE_STORE_STAGING,
    PHASE_COMMUNICATOR_INIT,
    PHASE_MODEL_BUILD,
    PHASE_EXTENSION_PRELOAD,
    PHASE_MODEL_EXTENSION_PRELOAD,
    PHASE_DECODE_SHARD_BUILD,
    PHASE_PREFILL_CONFIGURE,
    PHASE_PREWARM,
)

_PHASE_INDEX: Dict[str, int] = {
    name: index for index, name in enumerate(STARTUP_PHASES)
}

# Budgets are generous on purpose: this watchdog exists to end a startup that
# has STOPPED, not to police a slow one. A phase that is merely slow and
# progressing re-enters the record and gets its budget again.
#
# The default covers the long I/O- and build-bound phases, which on a large
# multi-node checkpoint are minutes each and scale with the store size.
DEFAULT_PHASE_BUDGET_S = 1800.0
#: Phases whose budget differs from :data:`DEFAULT_PHASE_BUDGET_S`.
PHASE_BUDGETS_S: Dict[str, float] = {
    # A rendezvous either completes for every rank together or never: once
    # every rank has reached it, it costs seconds. A budget far above that is
    # still well under the time a stalled one would otherwise burn.
    PHASE_COMMUNICATOR_INIT: 900.0,
}
#: How often the watchdog reads the record. Phase budgets are minutes, so a
#: coarse poll costs nothing and keeps the thread off a busy startup's back.
DEFAULT_POLL_INTERVAL_S = 5.0


def phase_budget_s(
    phase: str,
    *,
    scale: float = 1.0,
    budgets: Optional[Dict[str, float]] = None,
    default_budget_s: float = DEFAULT_PHASE_BUDGET_S,
) -> float:
    """``phase``'s budget in seconds, scaled by ``scale``."""
    if scale <= 0.0:
        raise ValueError(f"startup phase timeout scale must be positive, got {scale}")
    table = PHASE_BUDGETS_S if budgets is None else budgets
    return table.get(phase, default_budget_s) * scale


def _phase_index(phase: str) -> int:
    index = _PHASE_INDEX.get(phase)
    if index is None:
        raise ValueError(
            f"unknown startup phase {phase!r}; declare it in "
            "batchgen.worker.startup_phases.STARTUP_PHASES before reporting it"
        )
    return index


class StartupPhaseRecord:
    """Each process's current startup phase, in shared memory, without locks.

    One slot per process: slot 0 is the server itself, which runs the phases
    that precede the workers, and the workers take the slots after it. A slot
    is either idle or holds (phase, entry time).
    """

    #: Per slot: sequence counter, phase index, entry time.
    _FIELDS = 3
    #: Phase index of a slot that is in no tracked phase.
    _IDLE = -1

    def __init__(
        self,
        slots: int,
        array: Any,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if slots <= 0:
            raise ValueError(f"a startup phase record needs at least one slot, got {slots}")
        if len(array) != slots * self._FIELDS:
            raise ValueError(
                f"a {slots}-slot startup phase record needs "
                f"{slots * self._FIELDS} array entries, got {len(array)}"
            )
        self.slots = slots
        self._array = array
        self._clock = clock
        self.reset()

    @classmethod
    def create(
        cls,
        slots: int,
        ctx: Any,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> "StartupPhaseRecord":
        """A record in ``ctx``'s shared memory, inheritable by spawned workers.

        ``RawArray`` -- shared memory with no lock and no wake handshake -- for
        the reason given in this module's docstring and in
        ``WorkerManager._create_worker_channels``.
        """
        return cls(slots, ctx.RawArray("d", slots * cls._FIELDS), clock=clock)

    def _base(self, slot: int) -> int:
        if not 0 <= slot < self.slots:
            raise IndexError(
                f"startup phase slot {slot} is outside [0, {self.slots})"
            )
        return slot * self._FIELDS

    def _store(self, slot: int, index: int, entered_at: float) -> None:
        base = self._base(slot)
        array = self._array
        # Odd while the pair below is being written, even once it is stable.
        array[base] = array[base] + 1.0
        array[base + 1] = float(index)
        array[base + 2] = entered_at
        array[base] = array[base] + 1.0

    def reset(self, slots: Optional[Sequence[int]] = None) -> None:
        """Mark ``slots`` (every slot by default) idle.

        A worker cohort is replaced without replacing this record, so the next
        cohort's slots must not still hold the previous one's phases.
        """
        for slot in range(self.slots) if slots is None else slots:
            self._store(slot, self._IDLE, 0.0)

    def enter(self, slot: int, phase: str) -> None:
        """``slot`` has just entered ``phase``; its budget starts now."""
        self._store(slot, _phase_index(phase), self._clock())

    def leave(self, slot: int) -> None:
        """``slot`` is in no tracked phase, so no budget applies to it."""
        self._store(slot, self._IDLE, 0.0)

    def read(self, slot: int) -> Optional[Tuple[str, float]]:
        """``(phase, seconds in it)`` for ``slot``.

        None when the slot is idle, or when the pair was being written while
        it was read -- the next poll reads it whole, and a phase budget is
        minutes, so skipping one poll of a slot costs nothing.
        """
        base = self._base(slot)
        array = self._array
        before = array[base]
        if before % 2.0 != 0.0:
            return None
        index = int(array[base + 1])
        entered_at = array[base + 2]
        if array[base] != before:
            return None
        if not 0 <= index < len(STARTUP_PHASES):
            return None
        return STARTUP_PHASES[index], max(0.0, self._clock() - entered_at)

    def reporter(self, slot: int) -> "StartupPhaseReporter":
        """The handle the process owning ``slot`` reports through."""
        return StartupPhaseReporter(self, slot)


class StartupPhaseReporter:
    """One process's own slot in a :class:`StartupPhaseRecord`.

    Handed to the process that owns the slot so it never needs the record's
    shape, exactly as ``BatchGenWorker`` receives a cohort-shutdown probe
    rather than the shared flag behind it.
    """

    def __init__(self, record: StartupPhaseRecord, slot: int) -> None:
        self._record = record
        self._slot = slot

    @property
    def slot(self) -> int:
        return self._slot

    def enter(self, phase: str) -> None:
        self._record.enter(self._slot, phase)

    def leave(self) -> None:
        self._record.leave(self._slot)

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        """Report ``name`` for the body's duration, then go idle.

        Phases are wrapped back to back along a startup path, so the idle gaps
        between them are the few glue statements that separate the calls. An
        idle slot has no budget, which is what lets the same record cover a
        process whose remaining startup work is not phase-tracked.
        """
        self.enter(name)
        try:
            yield
        finally:
            self.leave()


class NullStartupPhaseReporter:
    """Reports nothing: the default for a worker no server is watching.

    A worker started outside the managed server path (tests, the legacy
    launcher) has no slot, and every phase-wrapped step must still run.
    """

    slot = -1

    def enter(self, phase: str) -> None:
        # Still validated, so an undeclared phase name is a hard error
        # wherever it is reported, not only under a watching server.
        _phase_index(phase)

    def leave(self) -> None:
        return

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        self.enter(name)
        try:
            yield
        finally:
            self.leave()


#: Shared singleton: the null reporter holds no state.
NULL_STARTUP_PHASE_REPORTER = NullStartupPhaseReporter()


class StartupPhaseWatchdog:
    """Fails a startup that has stopped inside one phase, with the evidence.

    Reads every slot of a record on an interval. The first poll that finds
    any slot past its phase's budget logs one CRITICAL line per stuck slot --
    naming the slot, the phase and how long it has been in it -- and only
    then calls ``on_breach`` with a reason naming the same, which the server
    turns into its ordinary worker-fatal startup failure. Logging first is
    the point: the failure must leave behind WHERE it was stuck, which is
    exactly what a silent spin or a killed launch does not.

    It fires at most once: the failure it hands to ``on_breach`` ends the
    startup, and repeating it would only race that teardown.
    """

    def __init__(
        self,
        record: StartupPhaseRecord,
        slot_labels: Sequence[str],
        on_breach: Callable[[str], None],
        *,
        scale: float = 1.0,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        budgets: Optional[Dict[str, float]] = None,
        default_budget_s: float = DEFAULT_PHASE_BUDGET_S,
        log: Optional[logging.Logger] = None,
    ) -> None:
        if len(slot_labels) != record.slots:
            raise ValueError(
                f"a {record.slots}-slot record needs {record.slots} slot "
                f"labels, got {len(slot_labels)}"
            )
        if scale <= 0.0:
            raise ValueError(
                f"startup phase timeout scale must be positive, got {scale}"
            )
        self._record = record
        self._slot_labels = tuple(slot_labels)
        self._on_breach = on_breach
        self._scale = float(scale)
        self._poll_interval_s = poll_interval_s
        self._budgets = budgets
        self._default_budget_s = default_budget_s
        self._log = log or logger
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fired = False

    @property
    def fired(self) -> bool:
        """Has a budget been breached and reported?"""
        return self._fired

    def budget_s(self, phase: str) -> float:
        return phase_budget_s(
            phase,
            scale=self._scale,
            budgets=self._budgets,
            default_budget_s=self._default_budget_s,
        )

    def snapshot(self) -> List[Tuple[str, str, float]]:
        """``(label, phase, seconds)`` for every slot in a tracked phase."""
        live: List[Tuple[str, str, float]] = []
        for slot, label in enumerate(self._slot_labels):
            current = self._record.read(slot)
            if current is None:
                continue
            live.append((label, current[0], current[1]))
        return live

    def poll(self) -> Optional[str]:
        """One pass over the record; the breach reason, or None."""
        stuck = [
            (label, phase, seconds, self.budget_s(phase))
            for label, phase, seconds in self.snapshot()
            if seconds > self.budget_s(phase)
        ]
        if not stuck:
            return None
        for label, phase, seconds, budget in stuck:
            self._log.critical(
                "[startup-watchdog] %s has been in startup phase %r for "
                "%.0fs, past its %.0fs budget; failing this startup so the "
                "whole cohort aborts instead of waiting on it further",
                label,
                phase,
                seconds,
                budget,
            )
        return "startup phase budget exceeded: " + "; ".join(
            f"{label} stuck in {phase} for {seconds:.0f}s "
            f"(budget {budget:.0f}s)"
            for label, phase, seconds, budget in stuck
        )

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="startup-phase-watchdog",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop_event.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout_s)

    def _run(self) -> None:
        while not self._stop_event.wait(self._poll_interval_s):
            try:
                reason = self.poll()
            except Exception:
                self._log.exception(
                    "The startup phase watchdog could not read the phase record"
                )
                continue
            if reason is None:
                continue
            self._fired = True
            try:
                self._on_breach(reason)
            except Exception:
                self._log.exception(
                    "Failing the startup after a startup phase budget breach "
                    "did not complete: %s",
                    reason,
                )
            return


__all__ = [
    "DEFAULT_PHASE_BUDGET_S",
    "DEFAULT_POLL_INTERVAL_S",
    "NULL_STARTUP_PHASE_REPORTER",
    "NullStartupPhaseReporter",
    "PHASE_BUDGETS_S",
    "PHASE_COMMUNICATOR_INIT",
    "PHASE_DECODE_SHARD_BUILD",
    "PHASE_EXTENSION_PRELOAD",
    "PHASE_MODEL_BUILD",
    "PHASE_MODEL_EXTENSION_PRELOAD",
    "PHASE_PREFILL_CONFIGURE",
    "PHASE_PREWARM",
    "PHASE_STORE_STAGING",
    "STARTUP_PHASES",
    "StartupPhaseRecord",
    "StartupPhaseReporter",
    "StartupPhaseWatchdog",
    "phase_budget_s",
]
