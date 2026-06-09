# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
# ---------------------------------------------------------------------------- #

"""DeepSeek-V4-Flash parallel strategy manager.

This manager is V4-native and self-contained.  It instantiates the V4 model
surface, loads skeleton tensors by remapping BatchGen attribute names to V4
checkpoint names, and attaches local V4 wrappers for DP attention + EP MoE.
"""

from __future__ import annotations

import logging
import os
import time

import torch

from .model import DeepSeekV4FlashForCausalLM
from .tensor_contract import (
    build_v4_weight_contract,
    model_key_to_checkpoint_key,
)
from .wrappers import DeepSeekV4FlashAttnWrapper, DeepSeekV4FlashExpertWrapper


class DeepSeekV4FlashParallelStrategyManager:
    def __init__(
        self,
        loaded_model_config,
        engine_config,
        model_config,
        core_engine,
        skeleton_state_dict,
        local_rank,
        global_rank,
        world_size,
    ):
        self.loaded_model_config = loaded_model_config
        self.engine_config = engine_config
        self.model_config = model_config
        self.core_engine = core_engine
        self.skeleton_state_dict = skeleton_state_dict
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.rank = global_rank
        self.state_dict_name_map, self.weight_copy_task = (
            build_v4_weight_contract(model_config)
        )

    def configure_prefill(self):
        if self.loaded_model_config is not None:
            self.loaded_model_config.phase = "prefill"
        start = time.perf_counter()
        self.model = DeepSeekV4FlashForCausalLM(self.loaded_model_config)
        self._load_model_skeleton()
        self._configure_moe_ranges(prefill=True, comm=None)
        self._config_attn_module()
        self._config_expert_module()
        self._config_lm_head_hook()
        self.model.eval()
        self.model.to(self.engine_config.Basic_Config.device_torch)
        if self.rank == 0:
            logging.info(
                "DeepSeek-V4-Flash prefill configured in %.2fs",
                time.perf_counter() - start,
            )
        return self.model, self.weight_copy_task

    def configure_decoding(self, padding_bsz=None, comm=None):
        if self.loaded_model_config is not None:
            self.loaded_model_config.phase = "decode"
        start = time.perf_counter()
        self.model = DeepSeekV4FlashForCausalLM(self.loaded_model_config)
        self._load_model_skeleton()
        self._configure_moe_ranges(prefill=False, comm=comm)
        effective_padding_bsz = padding_bsz if padding_bsz is not None else 128
        self._init_decoding_padding_bsz(effective_padding_bsz)
        self._mark_local_experts_persistent()
        self._config_attn_module()
        self._config_expert_module()
        self._load_local_routed_experts()
        self._config_lm_head_hook()
        self.model.eval()
        self.model.to(self.engine_config.Basic_Config.device_torch)
        if self.rank == 0:
            logging.info(
                "DeepSeek-V4-Flash decode configured in %.2fs",
                time.perf_counter() - start,
            )
        return self.model, self.weight_copy_task

    def _init_decoding_padding_bsz(self, padding_bsz):
        env_max_bsz = os.getenv("BATCHGEN_MAX_RANK_BSZ")
        max_rank_bsz = int(env_max_bsz) if env_max_bsz else int(padding_bsz)
        if self.rank == 0:
            logging.info(
                "[DECODE] DeepSeek-V4 padding batch size: %s%s",
                max_rank_bsz,
                " (from BATCHGEN_MAX_RANK_BSZ)" if env_max_bsz else "",
            )
        for layer in self.model.model.layers:
            layer.mlp.init_num_tokens(max_rank_bsz)

    def _grouped_moe_enabled(self) -> bool:
        return os.environ.get("BATCHGEN_V4_GROUPED_MOE", "0") == "1"

    def _local_routed_expert_keys(self):
        keys = []
        for layer in self.model.model.layers:
            mlp = layer.mlp
            layer_idx = mlp.layer_idx
            for e in range(
                mlp.routed_expert_start_idx, mlp.routed_expert_end_idx
            ):
                keys.append((layer_idx, e, f"routed_expert_{layer_idx}_{e}"))
        return keys

    def _mark_local_experts_persistent(self) -> None:
        # Grouped MoE needs owned experts resident (not streamed through the
        # rolling buffer pool), mirroring GLM5/DeepSeek-V3. Remove them from the
        # weight-copy (streaming) task so _config_expert_module marks them
        # persistent. Gated: default path keeps all experts streamed.
        if not self._grouped_moe_enabled():
            return
        local = {k for _, _, k in self._local_routed_expert_keys()}
        self.weight_copy_task["routed_expert"] = [
            k
            for k in self.weight_copy_task.get("routed_expert", [])
            if k not in local
        ]

    def _load_local_routed_experts(self) -> None:
        # Load persistent owned-expert weights resident from the host parameter
        # store via core_engine.get_tensor (stable, not the recyclable get_weights
        # buffer pool), mirroring GLM5._load_local_routed_experts.
        if not self._grouped_moe_enabled():
            return
        device = self.engine_config.Basic_Config.device_torch
        resident_bytes = 0
        for layer_idx, expert_idx, key in self._local_routed_expert_keys():
            tensors = self.core_engine.get_tensor(key)
            moved = {k: v.to(device) for k, v in tensors.items()}
            for v in moved.values():
                if v.is_cuda:
                    resident_bytes += v.numel() * v.element_size()
            placeholder = (
                self.model.model.layers[layer_idx]
                .mlp.experts[expert_idx]
                .module
            )
            placeholder.set_runtime_tensors(moved)
        if self.rank == 0:
            logging.info(
                "[V4 GROUPED] persistent expert resident bytes: %.2f GiB",
                resident_bytes / 1024**3,
            )
            placeholder.set_runtime_tensors(
                {k: v.to(device) for k, v in tensors.items()}
            )

    def set_num_tokens_per_rank(self, num_tokens_per_rank):
        for layer in self.model.model.layers:
            layer.mlp.set_num_tokens_per_rank(int(num_tokens_per_rank))

    def _load_model_skeleton(self):
        loaded = 0
        skipped = 0
        missing = []
        for key, param in self.model.named_parameters():
            ckpt_key = model_key_to_checkpoint_key(key)
            if ckpt_key in self.state_dict_name_map:
                skipped += 1
                continue
            if ckpt_key in self.skeleton_state_dict:
                param.data = self.skeleton_state_dict[ckpt_key]
                loaded += 1
            else:
                missing.append((key, ckpt_key))
        if self.rank == 0:
            logging.info(
                "DeepSeek-V4 skeleton loaded=%d skipped_bundle=%d missing=%d",
                loaded,
                skipped,
                len(missing),
            )
            if missing:
                logging.warning(
                    "DeepSeek-V4 missing skeleton samples: %s", missing[:20]
                )

    def _configure_moe_ranges(self, prefill: bool, comm) -> None:
        if prefill:
            rank = 0
            world_size = 1
        else:
            rank = self.global_rank
            world_size = self.world_size
        for layer in self.model.model.layers:
            layer.mlp.configure_ep(rank, world_size, comm=comm)

    def _is_attn_in_weight_copy_task(self, module_key: str) -> bool:
        for task_type, keys in self.weight_copy_task.items():
            if task_type.startswith("attn") and module_key in keys:
                return True
        return False

    def _config_attn_module(self) -> None:
        for layer_idx, layer in enumerate(self.model.model.layers):
            persistent = not self._is_attn_in_weight_copy_task(
                f"attn_{layer_idx}"
            )
            layer.self_attn = DeepSeekV4FlashAttnWrapper(
                layer.self_attn,
                layer_idx,
                self.core_engine,
                self.engine_config,
                self.model_config,
                persistent=persistent,
            )
            layer.attn = layer.self_attn

    def _config_expert_module(self) -> None:
        for layer_idx, layer in enumerate(self.model.model.layers):
            shared_key = f"shared_expert_{layer_idx}"
            shared_persistent = shared_key not in self.weight_copy_task.get(
                "shared_expert", []
            )
            layer.mlp.shared_experts = DeepSeekV4FlashExpertWrapper(
                layer.mlp.shared_experts,
                layer_idx,
                -1,
                self.core_engine,
                self.engine_config,
                self.model_config,
                persistent=shared_persistent,
            )
            for expert_idx, expert in enumerate(layer.mlp.experts):
                routed_key = f"routed_expert_{layer_idx}_{expert_idx}"
                persistent = routed_key not in self.weight_copy_task.get(
                    "routed_expert", []
                )
                layer.mlp.experts[expert_idx] = DeepSeekV4FlashExpertWrapper(
                    expert,
                    layer_idx,
                    expert_idx,
                    self.core_engine,
                    self.engine_config,
                    self.model_config,
                    persistent=persistent,
                )

    def _lm_head_forward_pre_hook(self, module, input):
        del module
        return input[0][:, -1, :].unsqueeze(1)

    def _config_lm_head_hook(self) -> None:
        self.model.lm_head.register_forward_pre_hook(
            self._lm_head_forward_pre_hook
        )
