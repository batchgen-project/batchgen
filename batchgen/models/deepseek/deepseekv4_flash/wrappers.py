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

import os
import time
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from batchgen.models.wrappers import AttnWrapperBase, ExpertWrapperBase


def _v4_diag(message: str) -> None:
    if os.environ.get("BATCHGEN_V4_DIAG", "0") != "1":
        return
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
    print(f"[V4-DIAG rank={rank}] {message}", flush=True)


def _v4_timing_enabled() -> bool:
    return os.environ.get("BATCHGEN_V4_TIMING", "0") == "1"


def _v4_timing(message: str) -> None:
    if not _v4_timing_enabled():
        return
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
    print(f"[V4-TIMING rank={rank}] {message}", flush=True)


def _v4_sync_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


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

    def _gather_paged_kv(
        self,
        gpu_paged_kv_manager,
        cache_seqlens: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if batch_size == 0:
            return torch.empty(
                0,
                0,
                self.module.head_dim,
                dtype=self.module.attn_sink.dtype,
                device=self.module.attn_sink.device,
            )
        max_len = int(cache_seqlens.max().item())
        page_size = int(gpu_paged_kv_manager.config.page_size_tokens)
        num_pages = (max_len + page_size - 1) // page_size
        k_cache, _, page_table = gpu_paged_kv_manager.get_layer_kv_with_page_table(
            self.layer_idx
        )
        table = page_table[:batch_size, :num_pages].to(device=k_cache.device)
        valid_pages = table >= 0
        gathered = k_cache[table.clamp_min(0).long()]
        gathered = gathered.masked_fill(
            ~valid_pages[:, :, None, None, None],
            0,
        )
        kv = gathered.reshape(batch_size, num_pages * page_size, *gathered.shape[3:])
        kv = kv[:, :max_len]
        if kv.dim() == 4:
            if kv.size(2) != 1:
                raise RuntimeError(
                    "DeepSeek-V4 paged KV fallback expects one KV head, "
                    f"got shape={tuple(kv.shape)}"
                )
            kv = kv.squeeze(2)
        return kv.contiguous()

    def _append_decode_kv(
        self,
        present: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        gpu_paged_kv_manager,
    ) -> None:
        if present is None or present.numel() == 0:
            return
        k_tensor = present
        if k_tensor.dim() == 3:
            k_tensor = k_tensor.unsqueeze(2)
        if position_ids is not None and gpu_paged_kv_manager is not None:
            sequence_lengths = position_ids.squeeze(-1).to(
                device=gpu_paged_kv_manager.device,
                dtype=torch.int32,
            )
            gpu_paged_kv_manager.update_layer_decode_new_token(
                k_tensor=k_tensor.to(gpu_paged_kv_manager.device),
                v_tensor=None,
                sequence_lengths=sequence_lengths,
                layer_idx=self.layer_idx,
            )
        kv_append_callback = getattr(AttnWrapperBase, "kv_append_callback", None)
        if kv_append_callback is not None:
            kv_append_callback(self.layer_idx, k_tensor, None)

    def _offload_prepacked_kv(self, offload_kv: torch.Tensor) -> None:
        host_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)
        if host_view is None:
            host_view = AttnWrapperBase.host_paged_kv_worker_view
        if host_view is None or AttnWrapperBase.cur_batch is None:
            return
        cu_seqlens = AttnWrapperBase.prepack_cu_seqlens
        num_sequences = AttnWrapperBase.prepack_num_sequences
        if cu_seqlens is None or num_sequences is None:
            raise RuntimeError("DeepSeek-V4 prepacked KV offload is missing cu_seqlens.")
        cu_seqlens = cu_seqlens.to(device="cpu", dtype=torch.long)
        for seq_idx in range(int(num_sequences)):
            start = int(cu_seqlens[seq_idx].item())
            end = int(cu_seqlens[seq_idx + 1].item())
            seq_len = end - start
            seq_kv = offload_kv[start:end].unsqueeze(0).unsqueeze(2)
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

    def _forward_prepacked(self, hidden_states: torch.Tensor):
        cu_seqlens = AttnWrapperBase.prepack_cu_seqlens
        num_sequences = AttnWrapperBase.prepack_num_sequences
        if cu_seqlens is None or num_sequences is None:
            raise RuntimeError("DeepSeek-V4 prepacked attention is missing cu_seqlens.")
        cu_cpu = cu_seqlens.to(device="cpu", dtype=torch.long)
        output = torch.empty_like(hidden_states)
        offload_kv = hidden_states.new_empty(
            hidden_states.size(1),
            self.module.head_dim,
        )
        flat_position_ids = AttnWrapperBase.position_ids
        for seq_idx in range(int(num_sequences)):
            start = int(cu_cpu[seq_idx].item())
            end = int(cu_cpu[seq_idx + 1].item())
            seq_hidden = hidden_states[:, start:end, :]
            seq_position_ids = None
            if flat_position_ids is not None:
                seq_position_ids = flat_position_ids[start:end].to(
                    device=hidden_states.device,
                    dtype=torch.long,
                ).view(1, -1)
            seq_out, _, seq_kv = self.module(
                seq_hidden,
                attention_mask=None,
                position_ids=seq_position_ids,
                past_key_value=None,
                cache_seqlens=None,
                use_cache=False,
            )
            output[:, start:end, :] = seq_out
            offload_kv[start:end] = seq_kv.squeeze(0)
        self._offload_prepacked_kv(offload_kv)
        return output, None, None

    def forward(self, *args, **kwargs):
        timing = self.phase == "decode" and _v4_timing_enabled()
        last_time = _v4_sync_time() if timing else 0.0

        def mark(label: str) -> None:
            nonlocal last_time
            if not timing:
                return
            now = _v4_sync_time()
            _v4_timing(
                f"wrapper attn L{self.layer_idx} {label} "
                f"{(now - last_time) * 1000:.2f}ms"
            )
            last_time = now

        if self.phase == "decode":
            _v4_diag(f"wrapper attn L{self.layer_idx} decode enter")
            kwargs["position_ids"] = AttnWrapperBase.position_ids
            kwargs["cache_seqlens"] = AttnWrapperBase.cache_seqlens
            past_key_states = AttnWrapperBase.past_key_states
            gpu_paged_kv_manager = AttnWrapperBase.gpu_paged_kv_manager
            if past_key_states is not None:
                kwargs["past_key_value"] = past_key_states[self.layer_idx]
            elif gpu_paged_kv_manager is not None and args and args[0].shape[0] > 0:
                kwargs["past_key_value"] = self._gather_paged_kv(
                    gpu_paged_kv_manager,
                    AttnWrapperBase.cache_seqlens,
                    args[0].shape[0],
                )
            if args and isinstance(args[0], torch.Tensor) and args[0].shape[0] == 0:
                _v4_diag(f"wrapper attn L{self.layer_idx} empty skip")
                return self.module.empty_forward(args[0])
        if self.phase == "decode":
            _v4_diag(f"wrapper attn L{self.layer_idx} load enter")
        self._load_runtime_tensors()
        mark("load")
        try:
            if self.phase == "prefill" and AttnWrapperBase.prepack_mode:
                if not args or not isinstance(args[0], torch.Tensor):
                    raise RuntimeError("DeepSeek-V4 prepacked attention missing hidden states.")
                result = self._forward_prepacked(args[0])
                mark("forward")
                return result
            if self.phase == "decode":
                _v4_diag(f"wrapper attn L{self.layer_idx} forward enter")
            result = self.module(*args, **kwargs)
            if self.phase == "decode":
                _v4_diag(f"wrapper attn L{self.layer_idx} forward done")
            mark("forward")
            if self.phase == "prefill":
                self._offload_prefill_kv(result[2], kwargs.get("attention_mask"))
            elif self.phase == "decode":
                self._append_decode_kv(
                    result[2],
                    kwargs.get("position_ids"),
                    AttnWrapperBase.gpu_paged_kv_manager,
                )
            return result
        finally:
            if self.phase == "decode":
                _v4_diag(f"wrapper attn L{self.layer_idx} release enter")
            self._release_runtime_tensors()
            mark("release")
            if self.phase == "decode":
                _v4_diag(f"wrapper attn L{self.layer_idx} release done")

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
        timing = getattr(self, "phase", None) == "decode" and _v4_timing_enabled()
        last_time = _v4_sync_time() if timing else 0.0

        def mark(label: str) -> None:
            nonlocal last_time
            if not timing:
                return
            now = _v4_sync_time()
            _v4_timing(
                f"wrapper expert {self.module_key} {label} "
                f"{(now - last_time) * 1000:.2f}ms"
            )
            last_time = now

        if getattr(self, "phase", None) == "decode":
            _v4_diag(f"wrapper expert {self.module_key} load enter")
        self._load_runtime_tensors()
        mark("load")
        try:
            if getattr(self, "phase", None) == "decode":
                _v4_diag(f"wrapper expert {self.module_key} forward enter")
            result = self.module(*args, **kwargs)
            if getattr(self, "phase", None) == "decode":
                _v4_diag(f"wrapper expert {self.module_key} forward done")
            mark("forward")
            return result
        finally:
            if getattr(self, "phase", None) == "decode":
                _v4_diag(f"wrapper expert {self.module_key} release enter")
            self._release_runtime_tensors()
            mark("release")
            if getattr(self, "phase", None) == "decode":
                _v4_diag(f"wrapper expert {self.module_key} release done")
