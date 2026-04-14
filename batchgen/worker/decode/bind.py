"""Phase 2.8.2a — native wrapper around ``adapter.bind_decode_context``.

Legacy ``_decode_bind_attn_wrapper`` (batchgen_worker.py:7838-7894)
touches model-class infrastructure: it sets class-level fields on
``AttnWrapperBase`` / ``Attn_Wrapper`` and handles the
``DualKVCacheCoordinator`` split. POIS's control-flow rule is about
admission / watermark / state transitions — not about class-level
singleton binding — so the actual bind logic stays inside the
adapter (``LegacyWorkerBackend.bind_decode_context``), while this
module gives the native decode loop a named step to call.

Keeping the wrapper separate from the scheduler does two things:
  * makes the decode-loop skeleton (Stage 2h+2i) read linearly — one
    helper per logical step;
  * centralises the future migration of the bind into a native
    ``batchgen/worker/attn_wrapper_bind.py`` if we later decide to
    move the class-level mutations out of legacy (not in scope for
    Stage 2).
"""

from __future__ import annotations

from typing import Any

from batchgen.worker.protocols import LegacyInfraBackend


def bind_decode_context(
    adapter: LegacyInfraBackend,
    *,
    batch: list[int],
    past_key_states: Any = None,
    past_value_states: Any = None,
    scale_dict: dict | None = None,
) -> tuple[Any, Any]:
    """Bind ``AttnWrapperBase`` / ``Attn_Wrapper`` to the decode batch.

    Returns ``(gpu_manager, worker_view)`` — the handles the decode
    loop threads into :mod:`batchgen.worker.boundary` handler calls
    and downstream helpers.

    Raises:
        AssertionError: when the GPU page-table slot-to-seq-id order
            diverges from ``cur_batch`` at bind time. The adapter
            propagates the Phase 2.5 hard-fail intact — silent repair
            masks upstream allocation bugs.
    """
    return adapter.bind_decode_context(
        batch=batch,
        past_key_states=past_key_states,
        past_value_states=past_value_states,
        scale_dict=scale_dict,
    )


__all__ = ["bind_decode_context"]
