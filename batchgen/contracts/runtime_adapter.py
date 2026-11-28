"""Per-model runtime-behavior contract (modularization milestone M2).

Mirrors `batchgen/cuda_graph/adapter.py`: a frozen `RuntimeState` is the *sole*
coupling point between the generic runtime (`batchgen_worker.py`, `decode.py`,
`prefill.py`) and per-model behavior. The core must not branch on a model name
/ `model_type`; instead it calls the model's `ModelRuntimeAdapter`.

This absorbs the behavioral leaks catalogued in
`batchgen_design/core_model_purity_audit.md` (the `if "<model>" in model_type`
branches for attention-backend config, position-id computation, and per-token
KV byte sizing).

**Phase A (this commit):** land the contract only. The generic files do not call
it yet; per-model adapters + a dual-gated migration follow in Phase B/C
(see `batchgen_design/blackwell/..` style Phase A/B/C from the cuda-graph work).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import torch


class RuntimePhase(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class RuntimeState:
    """Snapshot the generic runtime passes to a `ModelRuntimeAdapter`.

    This is the ONLY coupling surface: adapters read these fields and nothing
    else (never reach into worker/model internals). Frozen so the invariant is
    enforceable.
    """

    phase: RuntimePhase
    attention_mask: Optional[torch.Tensor]
    max_input_length: int
    token_idx: int
    device: Optional[torch.device] = None


class ModelRuntimeAdapter(ABC):
    """Per-model runtime behaviors that previously leaked into the core as
    ``if "<model>" in model_type`` branches.

    Subclasses live in `batchgen/models/<org>/<model>/` and are returned by the
    model's initializer (``get_runtime_behavior_adapter()``). Only
    `past_kv_byte_size` is abstract — the other two have model-agnostic defaults
    that the common (GQA / non-MLA, no flash-attn toggle) case can use as-is.
    """

    def __init__(self, model_config: Any):
        self.model_config = model_config

    # --- attention backend (was decode.py:230, prefill.py:89/261) -----------
    def configure_attention_backend(self, model: Any, *, phase: RuntimePhase) -> None:
        """Default: no-op. Override to toggle e.g. ``_use_flash_attention_2``."""
        return

    # --- position ids (was decode.py:303-310 / 436-443) --------------------
    def compute_position_ids(self, state: RuntimeState) -> torch.Tensor:
        """Default: full ids in prefill, last-token id in decode.

        MLA models (e.g. DeepSeek) override to return full ids in decode too.
        """
        from batchgen.utils import create_position_ids_from_attention_mask

        pos = create_position_ids_from_attention_mask(state.attention_mask)
        if state.phase == RuntimePhase.PREFILL:
            return pos
        return pos[:, -1].unsqueeze(-1)

    # --- per-token KV byte size (was decode.py:376-411) --------------------
    @abstractmethod
    def past_kv_byte_size(self, state: RuntimeState) -> int:
        """Bytes of one token's KV for this model (model-specific cache layout).

        Computed from ``state.max_input_length + state.token_idx`` and the
        adapter's ``model_config`` dimensions. There is no universal default
        (the legacy code raised for unlisted models), so each model implements it.
        """


__all__ = ["RuntimePhase", "RuntimeState", "ModelRuntimeAdapter"]
