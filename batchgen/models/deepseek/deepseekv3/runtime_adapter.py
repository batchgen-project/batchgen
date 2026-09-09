"""DeepSeek-V3 (MLA) runtime-behavior adapter (M2 phase B).

Moves the `if "deepseek" in model_type` branches out of the generic runtime
(decode.py / prefill.py) into this per-model adapter. See
batchgen/contracts/runtime_adapter.py and
batchgen_design/core_model_purity_audit.md.
"""
from __future__ import annotations

import torch

from batchgen.contracts.runtime_adapter import (
    ModelRuntimeAdapter,
    RuntimePhase,
    RuntimeState,
)


class DeepseekV3RuntimeAdapter(ModelRuntimeAdapter):
    def configure_attention_backend(self, model, *, phase: RuntimePhase) -> None:
        # decode.py:230 set True in decode; prefill.py:89/261 set False in prefill.
        model.model._use_flash_attention_2 = (phase == RuntimePhase.DECODE)

    def compute_position_ids(self, state: RuntimeState) -> torch.Tensor:
        # DeepSeek uses FULL position ids in both phases (decode.py:303-310 deepseek
        # branch and :436-439), unlike the GQA default which slices the last token.
        from batchgen.utils import create_position_ids_from_attention_mask

        return create_position_ids_from_attention_mask(state.attention_mask)

    def past_kv_byte_size(self, state: RuntimeState) -> int:
        # decode.py:388-391 — MLA compressed KV; +1 token avoids a torch.cat in the
        # attention forward.
        return (
            (state.max_input_length + state.token_idx + 1)
            * self.model_config.compressed_kv_dim
        )
