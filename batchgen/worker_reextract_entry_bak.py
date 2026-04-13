"""LEGACY shim — preserved as ``worker_reextract_entry_bak.py``.

Phase-2 F9 moved orchestrator construction into
:meth:`batchgen.worker.orchestrator.WorkerOrchestrator.from_legacy_worker`.
This module is kept only as a back-compat re-export; new code should
import directly:

    from batchgen.worker.orchestrator import WorkerOrchestrator
    orch = WorkerOrchestrator.from_legacy_worker(worker)

The names below are aliases for the few external sites that still
import ``from batchgen.worker_reextract_entry_bak import ...``. Once
those sites move to the canonical import this module can be deleted.
"""

from __future__ import annotations

from batchgen.worker.orchestrator import WorkerOrchestrator


def should_use_reextract() -> bool:
    """DEPRECATED — Phase-2 retired the env-var gate. Always returns True."""
    return True


def build_orchestrator(worker: object) -> WorkerOrchestrator:
    """DEPRECATED — call ``WorkerOrchestrator.from_legacy_worker(worker)``."""
    return WorkerOrchestrator.from_legacy_worker(worker)


__all__ = ["should_use_reextract", "build_orchestrator"]
