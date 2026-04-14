"""Phase 2.8.2d — thin native wrapper around ``adapter.forward_decode_step``.

The actual forward pass (cache_seqlens construction, CUDA graph launch,
per-layer KV append callbacks, MoE expert routing, token sampling)
stays on the adapter. Legacy ``_decode_forward_step``
(batchgen_worker.py:8130-8376) is ~247 LOC of infrastructure-heavy
code tied to model-class state. POIS's control-flow rule is about
admission / watermark / state transitions, not about per-step GPU
primitives — so the native decode loop calls through the adapter for
this one step.

Keeping the wrapper separate from the scheduler mirrors the pattern
of :mod:`decode.bind`: the decode-loop skeleton reads as a linear
sequence of named helpers rather than raw adapter calls.
"""

from __future__ import annotations

from typing import Any

from batchgen.worker.protocols import LegacyInfraBackend


def forward_decode_step(
    adapter: LegacyInfraBackend,
    *,
    batch: list[int],
    new_tokens: Any,
    gpu_manager: Any,
    page_table_verified: bool,
    local_iteration: int,
) -> Any:
    """Run one decode forward pass + sample the next token.

    Returns the sampled ``new_tokens`` tensor (shape
    ``(batch, 1)``) for the next iteration. The adapter handles CTX
    invariant checks, page-table verification, cache_seqlens tensor
    construction, graph vs eager selection, and the per-layer
    callbacks that flush KV into the host paged store.
    """
    return adapter.forward_decode_step(
        batch=batch,
        new_tokens=new_tokens,
        gpu_manager=gpu_manager,
        page_table_verified=page_table_verified,
        local_iteration=local_iteration,
    )


__all__ = ["forward_decode_step"]
