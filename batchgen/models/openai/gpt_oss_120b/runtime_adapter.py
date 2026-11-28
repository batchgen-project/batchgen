"""GPT-OSS (GQA) runtime-behavior adapter (M2 phase B).

GPT-OSS uses the base defaults for position ids (last-token in decode) and
attention backend (no flash-attn toggle); only the per-token KV byte size
differs. See batchgen/contracts/runtime_adapter.py.
"""
from __future__ import annotations

from batchgen.contracts.runtime_adapter import ModelRuntimeAdapter, RuntimeState


class GptOssRuntimeAdapter(ModelRuntimeAdapter):
    def past_kv_byte_size(self, state: RuntimeState) -> int:
        # decode.py:400-407 — GQA: num_key_value_heads * head_dim, separate K and V.
        return (
            (state.max_input_length + state.token_idx + 1)
            * self.model_config.num_key_value_heads
            * self.model_config.head_dim
            * 2  # K + V
        )
