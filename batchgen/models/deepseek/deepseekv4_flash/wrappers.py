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

from typing import Dict

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
    ):
        super().__init__(module, layer_idx, core_engine, engine_config, model_config)
        self.persistent = persistent
        self.module_key = f"attn_{layer_idx}"

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
            result = self.module(*args, **kwargs)
            if self.phase == "prefill":
                self._offload_prefill_kv(result[2], kwargs.get("attention_mask"))
            return result
        finally:
            self._release_runtime_tensors()

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
