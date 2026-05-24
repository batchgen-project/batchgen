# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
# ---------------------------------------------------------------------------- #

"""DeepSeek-V4-Flash wrappers.

The wrappers are V4-local and intentionally do not import from other model
packages.  They bridge BatchGen's parameter-server lifecycle to the V4 model
slots defined in ``model.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from batchgen.models.wrappers import AttnWrapperBase, ExpertWrapperBase


class DeepSeekV4FlashAttnWrapper(AttnWrapperBase):
    def __init__(
        self,
        module: nn.Module,
        layer_idx: int,
        core_engine,
        engine_config,
        model_config,
        persistent: bool = False,
        v4_backend: Optional[Any] = None,
    ):
        super().__init__(
            module, layer_idx, core_engine, engine_config, model_config
        )
        self.persistent = persistent
        self.module_key = f"attn_{layer_idx}"
        self._v4_backend = v4_backend
        self._layer_config = None
        if v4_backend is not None:
            self._layer_config = v4_backend.layer_configs[layer_idx]

    def _load_runtime_tensors(self) -> None:
        if self.persistent:
            return
        tensors = self.load_weights(self.module_key)
        self.module.set_runtime_tensors(tensors)

    def _release_runtime_tensors(self) -> None:
        if self.persistent:
            return
        self.free_weights(self.module_key)
        self.module.clear_runtime_tensors()

    def forward(self, *args, **kwargs):
        if self.phase == "decode":
            kwargs["position_ids"] = AttnWrapperBase.position_ids
            kwargs["cache_seqlens"] = AttnWrapperBase.cache_seqlens
            past_key_states = AttnWrapperBase.past_key_states
            if past_key_states is not None:
                kwargs["past_key_value"] = past_key_states[self.layer_idx]
        self._load_runtime_tensors()
        try:
            if self.phase == "decode" and self._v4_backend is not None:
                return self._forward_decode_optimized(*args, **kwargs)
            result = self.module(*args, **kwargs)
            if self.phase == "prefill":
                self._offload_prefill_kv(
                    result[2], kwargs.get("attention_mask")
                )
            return result
        finally:
            self._release_runtime_tensors()

    def _forward_decode_optimized(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Any,
    ) -> tuple:
        """Optimized decode using V4 attention backend.

        Computes Q/KV via the module's projection layers, then delegates
        the attention mechanism to the backend (FlashMLA sparse/dense/compressed).
        """
        mod = self.module
        bsz, q_len, _ = hidden_states.shape

        # Q projection: hidden → wq_a → q_norm → wq_b → per-head RMSNorm
        q_low = mod.q_norm(mod.wq_a(hidden_states))
        q = mod.wq_b(q_low).view(bsz, q_len, mod.n_heads, mod.head_dim)
        q = q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + mod.eps)

        # KV projection: hidden → wkv → kv_norm
        kv = mod.kv_norm(mod.wkv(hidden_states))

        # Attention via backend (dispatches to FlashMLA sparse/dense/compressed)
        attn_output = self._v4_backend.forward(
            layer_config=self._layer_config,
            q=q.squeeze(1),  # decode: q_len==1 → [B, H, D]
            kv=kv.squeeze(1),  # decode: [B, D]
            attn_sink=mod.attn_sink,
            head_gates=kwargs.get("head_gates"),
        )

        # Output projection: wo_a → wo_b
        attn_output = attn_output.view(
            bsz,
            q_len,
            mod.o_groups,
            mod.n_heads // mod.o_groups * mod.head_dim,
        )
        from batchgen.models.deepseek.deepseekv4_flash.model import (
            _dequant_weight,
        )

        wo_a_weight = _dequant_weight(
            mod.wo_a.weight,
            mod.wo_a.scale,
            hidden_states.dtype,
        )
        wo_a = wo_a_weight.view(
            mod.o_groups,
            mod.o_lora_rank,
            mod.n_heads // mod.o_groups * mod.head_dim,
        )
        attn_output = torch.einsum("bsgd,grd->bsgr", attn_output, wo_a)
        attn_output = mod.wo_b(attn_output.flatten(2))
        return attn_output, None, kv

    def _offload_prefill_kv(
        self,
        offload_kv: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> None:
        host_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)
        if host_view is None:
            host_view = AttnWrapperBase.host_paged_kv_worker_view
        if host_view is None or AttnWrapperBase.cur_batch is None:
            return

        if attention_mask is None:
            attention_mask = AttnWrapperBase.attention_mask
        if attention_mask is None:
            seq_lens = [offload_kv.size(1)] * offload_kv.size(0)
        else:
            seq_lens = attention_mask.to(offload_kv.device).sum(dim=1).tolist()

        AttnWrapperBase.pending_prefill_offload_tensors.append(offload_kv)
        if offload_kv.is_cuda:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
            event.synchronize()

        for seq_idx, seq_len in enumerate(seq_lens):
            seq_len = int(seq_len)
            seq_kv = offload_kv[seq_idx : seq_idx + 1, :seq_len].unsqueeze(2)
            task = host_view.async_offload_layer_kv_to_host(
                layer_idx=self.layer_idx,
                sequence_ids=[AttnWrapperBase.cur_batch[seq_idx]],
                k_tensor=seq_kv,
                v_tensor=None,
                sequence_lengths=[seq_len],
            )
            AttnWrapperBase.pending_prefill_offload_tensors.append(seq_kv)
            if task is not None:
                AttnWrapperBase.pending_prefill_offload_tasks.append(task)


class DeepSeekV4FlashExpertWrapper(ExpertWrapperBase):
    def __init__(
        self,
        module: nn.Module,
        layer_idx: int,
        expert_idx: int,
        core_engine,
        engine_config,
        model_config,
        persistent: bool = False,
    ):
        super().__init__(
            module,
            layer_idx,
            expert_idx,
            core_engine,
            engine_config,
            model_config,
            persistent,
        )

    def _load_runtime_tensors(self) -> None:
        if self.persistent:
            return
        tensors: Dict[str, torch.Tensor] = self.load_weights(self.module_key)
        self.module.set_runtime_tensors(tensors)

    def _release_runtime_tensors(self) -> None:
        if self.persistent:
            return
        self.free_weights(self.module_key)
        self.module.clear_runtime_tensors()

    def forward(self, *args, **kwargs):
        self._load_runtime_tensors()
        try:
            return self.module(*args, **kwargs)
        finally:
            self._release_runtime_tensors()
