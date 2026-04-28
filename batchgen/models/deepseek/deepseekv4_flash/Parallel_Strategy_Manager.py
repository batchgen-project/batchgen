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
import time

import torch

from .model import DeepSeekV4FlashForCausalLM
from .tensor_contract import build_v4_weight_contract
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
        self.state_dict_name_map, self.weight_copy_task = build_v4_weight_contract(
            model_config
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
        del padding_bsz
        if self.loaded_model_config is not None:
            self.loaded_model_config.phase = "decode"
        start = time.perf_counter()
        self.model = DeepSeekV4FlashForCausalLM(self.loaded_model_config)
        self._load_model_skeleton()
        self._configure_moe_ranges(prefill=False, comm=comm)
        self._config_attn_module()
        self._config_expert_module()
        self._config_lm_head_hook()
        self.model.eval()
        self.model.to(self.engine_config.Basic_Config.device_torch)
        if self.rank == 0:
            logging.info(
                "DeepSeek-V4-Flash decode configured in %.2fs",
                time.perf_counter() - start,
            )
        return self.model, self.weight_copy_task

    def _model_key_to_ckpt_key(self, key: str) -> str:
        if key == "model.embed_tokens.weight":
            return "embed.weight"
        if key == "model.norm.weight":
            return "norm.weight"
        if key == "lm_head.weight":
            return "head.weight"
        if key.startswith("model.hc_head_"):
            return key.removeprefix("model.")
        if key.startswith("model.layers."):
            parts = key.split(".")
            layer_idx = parts[2]
            rest = ".".join(parts[3:])
            if rest.startswith("self_attn."):
                return f"layers.{layer_idx}.attn.{rest.removeprefix('self_attn.')}"
            if rest.startswith("attn."):
                return f"layers.{layer_idx}.attn.{rest.removeprefix('attn.')}"
            if rest.startswith("mlp.gate."):
                return f"layers.{layer_idx}.ffn.gate.{rest.removeprefix('mlp.gate.')}"
            if rest.startswith("ffn.gate."):
                return f"layers.{layer_idx}.ffn.gate.{rest.removeprefix('ffn.gate.')}"
            if rest.startswith("attn_norm."):
                return f"layers.{layer_idx}.attn_norm.{rest.removeprefix('attn_norm.')}"
            if rest.startswith("ffn_norm."):
                return f"layers.{layer_idx}.ffn_norm.{rest.removeprefix('ffn_norm.')}"
            if rest.startswith("hc_"):
                return f"layers.{layer_idx}.{rest}"
        return key

    def _load_model_skeleton(self):
        loaded = 0
        skipped = 0
        missing = []
        for key, param in self.model.named_parameters():
            ckpt_key = self._model_key_to_ckpt_key(key)
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
                logging.warning("DeepSeek-V4 missing skeleton samples: %s", missing[:20])

    def _configure_moe_ranges(self, prefill: bool, comm) -> None:
        if prefill:
            rank = 0
            world_size = 1
        else:
            rank = self.global_rank
            world_size = self.world_size
        for layer in self.model.model.layers:
            layer.mlp.configure_ep(rank, world_size, comm=comm)

    def _config_attn_module(self) -> None:
        for layer_idx, layer in enumerate(self.model.model.layers):
            persistent = f"attn_{layer_idx}" not in self.weight_copy_task.get("attn", [])
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
        self.model.lm_head.register_forward_pre_hook(self._lm_head_forward_pre_hook)
