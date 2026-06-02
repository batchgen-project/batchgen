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

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from batchgen.ckpt_converter.metadata_loader import resolve_torch_dtype
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

        converted_ckpt_dir = getattr(model_config, "converted_ckpt_dir", None)
        self._converted_ckpt_dir = (
            Path(converted_ckpt_dir) if converted_ckpt_dir is not None else None
        )
        self._prefill_full_tensors_cpu: Optional[Dict[str, torch.Tensor]] = None

    def set_v4_backend(self, backend) -> None:
        self._v4_backend = backend
        self._layer_config = (
            backend.layer_configs[self.layer_idx]
            if backend is not None
            else None
        )

    def _resolve_rank_ckpt_stem(self, rank: int) -> Path:
        if self._converted_ckpt_dir is None:
            raise RuntimeError(
                "DeepSeek-V4-Flash DP prefill requires "
                "model_config.converted_ckpt_dir to reconstruct full "
                "attention weights"
            )
        world_size = self.module.world_size
        for stem in (
            self._converted_ckpt_dir / f"model{rank}-mp{world_size}",
            self._converted_ckpt_dir / f"model{rank}",
        ):
            if (
                stem.with_suffix(".json").is_file()
                and stem.with_suffix(".bin").is_file()
            ):
                return stem
        for json_path in sorted(
            self._converted_ckpt_dir.glob(f"model{rank}*.json")
        ):
            stem = json_path.with_suffix("")
            if stem.with_suffix(".bin").is_file():
                return stem
        raise FileNotFoundError(
            f"No converted ckpt shard for rank={rank} under "
            f"{self._converted_ckpt_dir}"
        )

    @staticmethod
    def _read_tensors_from_shard(
        stem: Path, tensor_names: Tuple[str, ...]
    ) -> Dict[str, torch.Tensor]:
        with open(stem.with_suffix(".json")) as fh:
            meta = json.load(fh)["state_dict"]
        out: Dict[str, torch.Tensor] = {}
        with open(stem.with_suffix(".bin"), "rb") as fh:
            for name in tensor_names:
                entry = meta[name]
                fh.seek(int(entry["offset"]))
                raw = bytearray(fh.read(int(entry["byte_size"])))
                out[name] = (
                    torch.frombuffer(
                        raw, dtype=resolve_torch_dtype(str(entry["dtype"]))
                    )
                    .view(*entry["shape"])
                    .clone()
                )
        return out

    def _ensure_prefill_full_tensors_cpu(self) -> None:
        if self._prefill_full_tensors_cpu is not None:
            return
        prefix = f"layers.{self.layer_idx}.attn."
        names = [
            f"{prefix}wq_b.weight",
            f"{prefix}wq_b.scale",
            f"{prefix}wo_a.weight",
            f"{prefix}wo_b.weight",
            f"{prefix}wo_b.scale",
            f"{prefix}attn_sink",
        ]
        ratio = int(
            getattr(
                self._layer_config,
                "compress_ratio",
                getattr(self.module, "compress_ratio", 0),
            )
            or 0
        )
        has_c4_indexer = (
            ratio == 4 and getattr(self.module, "indexer", None) is not None
        )
        if has_c4_indexer:
            names.extend(
                [
                    f"{prefix}indexer.wq_b.weight",
                    f"{prefix}indexer.wq_b.scale",
                    f"{prefix}indexer.weights_proj.weight",
                ]
            )
        tensor_names = tuple(names)
        parts: Dict[str, List[torch.Tensor]] = {n: [] for n in names}
        for rank in range(self.module.world_size):
            shard = self._read_tensors_from_shard(
                self._resolve_rank_ckpt_stem(rank), tensor_names
            )
            for n in names:
                parts[n].append(shard[n])
        full_tensors = {
            "wq_b.weight": torch.cat(
                parts[f"{prefix}wq_b.weight"], dim=0
            ).contiguous(),
            "wq_b.scale": torch.cat(
                parts[f"{prefix}wq_b.scale"], dim=0
            ).contiguous(),
            "wo_a.weight": torch.cat(
                parts[f"{prefix}wo_a.weight"], dim=0
            ).contiguous(),
            "wo_b.weight": torch.cat(
                parts[f"{prefix}wo_b.weight"], dim=1
            ).contiguous(),
            "wo_b.scale": torch.cat(
                parts[f"{prefix}wo_b.scale"], dim=1
            ).contiguous(),
            "attn_sink": torch.cat(
                parts[f"{prefix}attn_sink"], dim=0
            ).contiguous(),
        }
        if has_c4_indexer:
            full_tensors.update(
                {
                    "indexer.wq_b.weight": torch.cat(
                        parts[f"{prefix}indexer.wq_b.weight"], dim=0
                    ).contiguous(),
                    "indexer.wq_b.scale": torch.cat(
                        parts[f"{prefix}indexer.wq_b.scale"], dim=0
                    ).contiguous(),
                    "indexer.weights_proj.weight": torch.cat(
                        parts[f"{prefix}indexer.weights_proj.weight"], dim=0
                    ).contiguous(),
                }
            )
        self._prefill_full_tensors_cpu = full_tensors

    def _load_runtime_tensors(self) -> None:
        if not self.persistent:
            tensors = self.load_weights(self.module_key)
            self.module.set_runtime_tensors(tensors)
        if self.module.world_size > 1:
            self._ensure_prefill_full_tensors_cpu()
            device = self.module.q_norm.weight.device
            self.module.set_prefill_full_tensors(
                {
                    name: tensor.to(device=device)
                    for name, tensor in self._prefill_full_tensors_cpu.items()
                }
            )

    def _release_runtime_tensors(self) -> None:
        if not self.persistent:
            self.free_weights(self.module_key)
            self.module.clear_runtime_tensors()
        self.module.clear_prefill_full_tensors()

    def forward(self, *args, **kwargs):
        self.module.runtime_phase = self.phase
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
                if self._is_v4_resident_prefill():
                    prefill_hidden = (
                        args[0] if args else kwargs.get("hidden_states")
                    )
                    self._populate_v4_prefill_kv(
                        result[2],
                        kwargs.get("attention_mask"),
                        prefill_hidden,
                    )
                else:
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

        # Padded DP rank with no real sequences: skip the V4 backend (its
        # per-step metadata is uninitialized) and return a zero attention output
        # so the collective MoE forward still completes; the result is discarded.
        if not AttnWrapperBase.cur_batch:
            kv_zero = mod.kv_norm(mod.wkv(hidden_states))
            return (
                torch.zeros_like(hidden_states),
                None,
                kv_zero,
            )

        # Decode is DP-attention: each rank uses the FULL head set + gathered
        # full wq_b/attn_sink (FlashMLA requires h_q>=64, not the local shard).
        dp_attention = mod.world_size > 1 and bool(
            getattr(mod, "_prefill_full_tensors", None)
        )
        n_attn_heads = mod.n_heads if dp_attention else mod.n_local_heads
        q_low = mod.q_norm(mod.wq_a(hidden_states))
        if dp_attention:
            from batchgen.models.deepseek.deepseekv4_flash.model import (
                _linear_from_weight,
            )

            q = _linear_from_weight(
                q_low,
                mod._get_prefill_full_tensor("wq_b.weight"),
                mod._prefill_full_tensors.get("wq_b.scale"),
            )
            attn_sink = mod._get_prefill_full_tensor("attn_sink")
        else:
            q = mod.wq_b(q_low)
            attn_sink = mod.attn_sink
        q = q.view(bsz, q_len, n_attn_heads, mod.head_dim)
        q = q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + mod.eps)

        # KV projection: hidden → wkv → kv_norm
        kv = mod.kv_norm(mod.wkv(hidden_states))

        dense_q = q.squeeze(1)
        dense_kv = kv.squeeze(1)
        backend_kwargs = {}
        ratio = (
            self._layer_config.compress_ratio
            if self._layer_config is not None
            else 0
        )
        if ratio == 4 and getattr(mod, "indexer", None) is not None:
            index_q, index_k, head_gates = self._v4_c4_indexer_inputs(
                q_low.squeeze(1), hidden_states.squeeze(1)
            )
            backend_kwargs.update(
                head_gates=head_gates,
                q_attn=dense_q,
                current_kv=dense_kv,
            )
            score_q, score_kv = index_q, index_k
        elif ratio == 128 and getattr(mod, "compressor", None) is not None:
            score_q, score_kv = dense_q, dense_kv
            backend_kwargs.update(
                compress_hidden_states=hidden_states.squeeze(1),
                compressor=self._runtime_kernel_compressor(
                    mod.compressor, rotate=False
                ),
                rope_cache=self._v4_compressed_rope_cache(hidden_states.device),
                current_kv=dense_kv,
            )
        else:
            score_q, score_kv = dense_q, dense_kv
            if "head_gates" in kwargs:
                backend_kwargs["head_gates"] = kwargs["head_gates"]

        # Attention via backend (dispatches to FlashMLA sparse/dense/compressed)
        attn_output = self._v4_backend.forward(
            layer_config=self._layer_config,
            q=score_q,
            kv=score_kv,
            attn_sink=attn_sink,
            **backend_kwargs,
        )

        from batchgen.models.deepseek.deepseekv4_flash.model import (
            _dequant_weight,
            _linear_from_weight,
        )

        n_groups = mod.o_groups if dp_attention else mod.n_local_groups
        attn_output = attn_output.view(
            bsz,
            q_len,
            n_groups,
            n_attn_heads // n_groups * mod.head_dim,
        )
        if dp_attention:
            wo_a_weight = _dequant_weight(
                mod._get_prefill_full_tensor("wo_a.weight"),
                None,
                hidden_states.dtype,
            )
            wo_a = wo_a_weight.view(
                n_groups,
                mod.o_lora_rank,
                n_attn_heads // n_groups * mod.head_dim,
            )
            attn_output = torch.einsum("bsgd,grd->bsgr", attn_output, wo_a)
            attn_output = _linear_from_weight(
                attn_output.flatten(2),
                mod._get_prefill_full_tensor("wo_b.weight"),
                mod._prefill_full_tensors.get("wo_b.scale"),
            )
            return attn_output, None, kv

        wo_a_weight = _dequant_weight(
            mod.wo_a.weight,
            mod.wo_a.scale,
            hidden_states.dtype,
        )
        wo_a = wo_a_weight.view(
            n_groups,
            mod.o_lora_rank,
            n_attn_heads // n_groups * mod.head_dim,
        )
        attn_output = torch.einsum("bsgd,grd->bsgr", attn_output, wo_a)
        attn_output = mod.wo_b(attn_output.flatten(2))
        return attn_output, None, kv

    def _v4_coordinator(self):
        manager = getattr(self.core_engine, "gpu_paged_kv_manager", None)
        from batchgen.kv_cache.deepseek_v4_kv_coordinator import (
            DeepSeekV4KVCoordinator,
        )

        if isinstance(manager, DeepSeekV4KVCoordinator):
            return manager
        return None

    def _is_v4_resident_prefill(self) -> bool:
        return self._v4_coordinator() is not None

    def _runtime_kernel_compressor(self, src, *, rotate: bool):
        cache = getattr(self, "_v4_kernel_compressors", None)
        if cache is None:
            cache = {}
            self._v4_kernel_compressors = cache
        key = id(src)
        comp = cache.get(key)
        if comp is None:
            from batchgen_kernels.attention.v4_compressor import (
                DeepSeekV4Compressor,
            )

            comp = DeepSeekV4Compressor(
                int(src.hidden_size),
                int(src.head_dim),
                int(src.rope_head_dim),
                int(src.compress_ratio),
                getattr(src.norm, "eps", 1e-6),
                overlap=bool(src.overlap),
                rotate=rotate,
                init_weights=False,
            ).to(src.ape.device)
            cache[key] = comp
        if src.wkv.weight is None or src.wgate.weight is None:
            raise RuntimeError(
                f"V4 compressor weights not loaded for layer {self.layer_idx}; "
                "cannot run compressed attention"
            )
        comp.ape = src.ape.detach().to(
            device=src.ape.device, dtype=torch.float32
        )
        comp.norm.weight = src.norm.weight.detach().to(
            device=src.norm.weight.device, dtype=torch.float32
        )
        comp.wkv_weight = src.wkv.weight.detach()
        comp.wgate_weight = src.wgate.weight.detach()
        comp.wkv_scale = (
            None
            if getattr(src.wkv, "scale", None) is None
            else src.wkv.scale.detach()
        )
        comp.wgate_scale = (
            None
            if getattr(src.wgate, "scale", None) is None
            else src.wgate.scale.detach()
        )
        return comp

    def _v4_prefill_rope_cache(self, device):
        cache = AttnWrapperBase.__dict__.get("_v4_prefill_rope_cache_cpu")
        if cache is None:
            from batchgen.attention.dsa.v4_flashmla_adapter import (
                build_v4_rope_cache,
            )

            rope_head_dim = int(
                getattr(self.model_config, "qk_rope_head_dim", 64)
            )
            theta = float(getattr(self.model_config, "rope_theta", 10000.0))
            max_pos = int(
                getattr(self.model_config, "max_position_embeddings", 8192)
            )
            cache = build_v4_rope_cache(
                max_pos=max_pos,
                theta=theta,
                rope_head_dim=rope_head_dim,
                device="cpu",
            )
            AttnWrapperBase._v4_prefill_rope_cache_cpu = cache
        return cache.to(device)

    def _v4_compress_rope_params(self):
        cfg = self.model_config
        scaling = getattr(cfg, "rope_scaling", None) or {}
        return dict(
            max_pos=int(getattr(cfg, "max_position_embeddings", 8192)),
            theta=float(getattr(cfg, "compress_rope_theta", 160000.0)),
            rope_head_dim=int(getattr(cfg, "qk_rope_head_dim", 64)),
            original_seq_len=int(
                scaling.get("original_max_position_embeddings", 0) or 0
            ),
            factor=float(scaling.get("factor", 1.0) or 1.0),
            beta_fast=float(scaling.get("beta_fast", 32.0)),
            beta_slow=float(scaling.get("beta_slow", 1.0)),
        )

    def _v4_compressed_cos_sin(self, device):
        tables = AttnWrapperBase.__dict__.get("_v4_compress_cos_sin_cpu")
        if tables is None:
            from batchgen.attention.dsa.v4_flashmla_adapter import (
                build_v4_rope_tables,
            )

            tables = build_v4_rope_tables(
                device="cpu", **self._v4_compress_rope_params()
            )
            AttnWrapperBase._v4_compress_cos_sin_cpu = tables
        cos_table, sin_table = tables
        return cos_table.to(device), sin_table.to(device)

    def _v4_compressed_rope_cache(self, device):
        cache = AttnWrapperBase.__dict__.get("_v4_compress_cos_sin_cache_cpu")
        if cache is None:
            from batchgen.attention.dsa.v4_flashmla_adapter import (
                build_v4_compress_cos_sin_cache,
            )

            cache = build_v4_compress_cos_sin_cache(
                device="cpu", **self._v4_compress_rope_params()
            )
            AttnWrapperBase._v4_compress_cos_sin_cache_cpu = cache
        return cache.to(device)

    def _v4_c4_indexer_inputs(self, q_low, hidden_states):
        from batchgen_kernels.attention.dsa.fused_indexer_score import (
            rope_hadamard_q,
        )
        from batchgen.models.deepseek.deepseekv4_flash.model import (
            _linear_from_weight,
        )

        mod = self.module
        idx = mod.indexer
        meta = self._v4_backend.metadata
        coordinator = meta.extras["coordinator"]
        sequence_ids = meta.extras["sequence_ids"]
        positions = meta.positions_casual
        seq_lens = meta.seq_lens_casual
        route = coordinator.get_layer_routing(self.layer_idx)
        device = hidden_states.device
        rope_dim = int(getattr(self.model_config, "qk_rope_head_dim", 64))

        bsz = hidden_states.shape[0]
        dp_attention = mod.world_size > 1 and bool(
            getattr(mod, "_prefill_full_tensors", None)
        )
        n_index_heads = (
            idx.n_heads if dp_attention else idx.n_heads // mod.world_size
        )
        if dp_attention:
            index_q = _linear_from_weight(
                q_low,
                mod._get_prefill_full_tensor("indexer.wq_b.weight"),
                mod._get_prefill_full_tensor("indexer.wq_b.scale"),
            )
        else:
            index_q = idx.wq_b(q_low)
        index_q = index_q.view(bsz, n_index_heads, idx.head_dim)
        cos_table, sin_table = self._v4_compressed_cos_sin(device)
        index_q = rope_hadamard_q(
            index_q, cos_table, sin_table, positions.to(torch.int64), rope_dim
        )

        seq_ids_list = (
            sequence_ids.tolist()
            if isinstance(sequence_ids, torch.Tensor)
            else list(sequence_ids)
        )
        seq_lens_list = seq_lens.tolist()
        ratio = 4
        max_clen = max(1, max(int(s) // ratio for s in seq_lens_list))
        index_k = torch.zeros(
            bsz, max_clen, idx.head_dim, device=device, dtype=torch.bfloat16
        )
        for b, seq_id in enumerate(seq_ids_list):
            clen = int(seq_lens_list[b]) // ratio
            if clen <= 0:
                continue
            cpos = torch.arange(clen, device=device, dtype=torch.long)
            slots = coordinator.indexer.sequence_token_slots(int(seq_id), cpos)
            k = coordinator.indexer.debug_read_indexer(
                layer_idx=route.indexer_layer_idx, token_slots=slots
            )
            index_k[b, :clen] = k

        softmax_scale = idx.head_dim**-0.5
        if dp_attention:
            head_gates = _linear_from_weight(
                hidden_states,
                mod._get_prefill_full_tensor("indexer.weights_proj.weight"),
                None,
            )
        else:
            head_gates = idx.weights_proj(hidden_states)
        head_gates = head_gates.view(bsz, n_index_heads)
        head_gates = head_gates * (softmax_scale * idx.n_heads**-0.5)
        return index_q, index_k, head_gates

    def _populate_v4_prefill_kv(
        self,
        prefill_kv: torch.Tensor,
        attention_mask: torch.Tensor | None,
        hidden_states: torch.Tensor | None = None,
    ) -> None:
        coordinator = self._v4_coordinator()
        if coordinator is None or AttnWrapperBase.cur_batch is None:
            return

        ratio = (
            self._layer_config.compress_ratio
            if self._layer_config is not None
            else 0
        )
        mod = self.module if ratio else None
        device = prefill_kv.device

        if attention_mask is None:
            attention_mask = AttnWrapperBase.attention_mask
        if attention_mask is None:
            seq_lens = [prefill_kv.size(1)] * prefill_kv.size(0)
        else:
            seq_lens = attention_mask.to(device).sum(dim=1).tolist()

        rope_cache = self._v4_prefill_rope_cache(device)
        compress_rope = (
            self._v4_compressed_rope_cache(device) if ratio else None
        )
        from batchgen.attention.dsa.v4_prefill_populate import (
            populate_v4_prefill_coordinator,
        )

        for seq_idx, seq_len in enumerate(seq_lens):
            seq_len = int(seq_len)
            if seq_len <= 0:
                continue
            sequence_id = int(AttnWrapperBase.cur_batch[seq_idx])
            coordinator.allocate_pages_for_sequences([sequence_id], [seq_len])
            swa_kv = prefill_kv[seq_idx, :seq_len]
            prompt_positions = torch.arange(
                seq_len, device=device, dtype=torch.long
            )

            c4_kv = indexer_k = None
            c128_hidden = None
            c128_compressor = None
            if ratio == 4 and hidden_states is not None:
                seq_hidden = hidden_states[seq_idx, :seq_len].float()
                main_comp = self._runtime_kernel_compressor(
                    mod.compressor, rotate=False
                )
                idx_comp = self._runtime_kernel_compressor(
                    mod.indexer.compressor, rotate=True
                )
                c4_kv = main_comp.forward_prefill(
                    seq_hidden, prompt_positions, compress_rope
                )
                indexer_k = idx_comp.forward_prefill(
                    seq_hidden, prompt_positions, compress_rope
                )
            elif ratio == 128 and hidden_states is not None:
                c128_hidden = hidden_states[seq_idx, :seq_len].float()
                c128_compressor = self._runtime_kernel_compressor(
                    mod.compressor, rotate=False
                )

            populate_v4_prefill_coordinator(
                coordinator=coordinator,
                layer_idx=self.layer_idx,
                sequence_id=sequence_id,
                prompt_positions=prompt_positions,
                swa_kv=swa_kv,
                rope_cache=rope_cache,
                c4_kv=c4_kv,
                indexer_k=indexer_k,
                c128_hidden_states=c128_hidden,
                compressor=c128_compressor,
                compress_rope_cache=compress_rope,
            )

            if ratio == 128 and c128_compressor is not None:
                remainder = seq_len % ratio
                if remainder > 0:
                    cutoff = seq_len - remainder
                    route = coordinator.get_layer_routing(self.layer_idx)
                    adapter = getattr(self._v4_backend, "_flashmla", None)
                    if adapter is not None and hasattr(
                        adapter, "seed_c128_decode_state"
                    ):
                        adapter.seed_c128_decode_state(
                            c128_layer_idx=route.c128_layer_idx,
                            sequence_id=sequence_id,
                            compressor=c128_compressor,
                            remainder_hidden=c128_hidden[cutoff:seq_len],
                            remainder_positions=prompt_positions[
                                cutoff:seq_len
                            ],
                        )

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

        target_kv_dtype = self.engine_config.Basic_Config.kv_dtype_torch
        if offload_kv.dtype != target_kv_dtype:
            offload_kv = offload_kv.to(target_kv_dtype)

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
