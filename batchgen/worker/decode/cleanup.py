"""Phase 2.8.2g — port of ``_decode_cleanup`` (8449-8493).

Drains deferred KV appends, waits on any still-pending async load,
unbinds the ``Attn_Wrapper`` / ``AttnWrapperBase`` class-level fields,
and disables the decode watchdog. Called from
``DecodeScheduler.run_continuous`` via a ``try/finally`` so it always
runs even when the loop exits via an exception.
"""

from __future__ import annotations

from batchgen.worker.boundary import BoundaryHandler
from batchgen.worker.protocols import LegacyInfraBackend


def decode_cleanup(
    adapter: LegacyInfraBackend,
    boundary: BoundaryHandler,
) -> None:
    """Tear down the decode context.

    Steps (mirror legacy 8451-8492):
      1. ``adapter.wait_pending_kv_append_tasks()`` — drain any
         deferred host-KV writes from the last forward step.
      2. ``adapter.wait_async_load_task(...)`` when the handler's
         pending stash has an un-integrated async load. Keeps the
         CUDA stream synchronous before the next phase touches the
         GPU KV manager.
      3. Clear the handler's pending stash so the next decode call
         starts with a clean slate.
      4. ``adapter.unbind_decode_context()`` — reset
         ``AttnWrapperBase`` class-level fields to ``None``. Prevents
         stale pointers from surviving into the next prefill / decode
         phase.
      5. ``adapter.disable_decode_watchdog()`` — matches legacy
         line 8492.

    Summary / cumulative-logging was in legacy 8476-8490 behind
    ``BATCHGEN_CB_DEBUG``. Native port leaves that to the
    DecodeScheduler if it decides to surface metrics; cleanup just
    tears down state.
    """
    adapter.wait_pending_kv_append_tasks()

    pending_task = boundary._pending_async_task
    if pending_task is not None:
        adapter.wait_async_load_task(pending_task)

    # Clear handler's async-load stash so the next decode phase starts
    # with a clean slate (no stray pointers to leaked handles).
    boundary._pending_async_task = None
    boundary._pending_load_uuids = []
    boundary._pending_load_local = []
    boundary._pending_load_global = []

    adapter.unbind_decode_context()
    adapter.disable_decode_watchdog()


__all__ = ["decode_cleanup"]
