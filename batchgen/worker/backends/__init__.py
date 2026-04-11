"""Real (production) backends satisfying the Protocols in worker/protocols.py.

Every backend here is a thin adapter — it wraps a production class
(``torch.distributed`` process group, ``GpuPagedKVCacheManager``,
``HostPagedKVWorkerView``, etc.) and forwards method calls through
without adding logic. The orchestrator wires these at startup on a
real GPU node; unit tests use the fakes in
``tests/unit/worker/fakes.py`` instead.

Design rules:
  - Every external import is LAZY (inside method bodies) so the
    modules load cleanly on a CPU-only jazz1 box. Only calling the
    methods forces the real torch / core_engine imports.
  - Adapters take the backing object via constructor; they never
    reach into module-level globals or os.environ.
  - Adapters are stateless w.r.t. the orchestrator — they do not
    remember uuids across calls. State lives in the WorkerState
    that handlers own.
"""

from __future__ import annotations

__all__: list[str] = []
