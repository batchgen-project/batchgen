# ---------------------------------------------------------------------------- #
#  MoE-Gen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#                                                                               #
#  you may obtain a copy of the license at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
#                                                                               #
#  unless required by applicable law or agreed to in writing, software          #
#  distributed under the license is distributed on an "as is" basis,            #
#  without warranties or conditions of any kind, either express or implied.     #
#  see the license for the specific language governing permissions and          #
#  limitations under the license.                                               #
# ---------------------------------------------------------------------------- #

import logging
import math
import os
import time
import types
from datetime import timedelta
from multiprocessing import Process
from typing import List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import triton.language as tl
from safetensors.torch import load_file
from tqdm import tqdm, trange

# from moe_gen.config import EngineConfig, ModelConfig
from ....config.config import EngineConfig, ModelConfig
from ....config.hf_config_parser import HuggingFaceModelConfig
from ...model_utils import ModelInitializer
from .configuration_qwen3_moe import Qwen3MoeConfig

# from transformers.models.qwen2_moe.modeling_qwen2_moe import repeat_kv

try:
    from moe_gen.core_engine import MoE_Gen as core_engine
except ImportError:
    # jit compile
    from core_engine import MoE_Gen as core_engine

from typing import Tuple

from moe_gen.models.Wrapper import Attn_Wrapper, Expert_Wrapper

from .modeling_qwen3_moe import (
    Qwen3MoeForCausalLM,
    apply_rotary_pos_emb,
    rotate_half,
)


class Qwen3Moe_Initializer(ModelInitializer):
    def __init__(
        self,
        huggingface_ckpt_name: str,
        hf_cache_dir: str,
        cache_dir: Optional[str],
        engine_config,
        skeleton_state_dict: Optional[dict],
        shm_name: str,
        tensor_meta_shm_name: str,
        pt_ckpt_dir,
        host_kv_cache_size: Optional[int] = None,
        local_rank: Optional[int] = 0,
        global_rank: Optional[int] = 0,
        world_size: Optional[int] = 1,
    ):
        self.huggingface_ckpt_name = huggingface_ckpt_name
        self.cache_dir = cache_dir
        self.pt_ckpt_dir = pt_ckpt_dir
        self.engine_config = engine_config
        self.skeleton_state_dict = skeleton_state_dict
        self.model = None
        self.hf_model_config = Qwen3MoeConfig.from_pretrained(
            self.cache_dir,
            # huggingface_ckpt_name,
            # cache_dir=hf_cache_dir,
            # trust_remote_code=True,
        )
        self.hf_model_config._name_or_path = huggingface_ckpt_name
        self.hf_model_config.architectures = ["Qwen3MoeForCausalLM"]

        self.host_kv_cache_size = host_kv_cache_size
        self.host_kv_cache_byte_size = host_kv_cache_size * (1024**3)

        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size

        # self.fp8_weights_IPC_handle = {}

        # TODO:
        self.model_config = self._parse_model_config()
        self._default_engine_config()
        self.state_dict_name_map = {}
        self.weight_copy_task = {}

        self.shm_name = shm_name
        self.tensor_meta_shm_name = tensor_meta_shm_name

    def _default_engine_config(self):
        props = torch.cuda.get_device_properties(
            self.engine_config.Basic_Config.device
        )
        total_memory = props.total_memory / (1024**3)
        logging.info(f"Current device total memory: {total_memory} GB")

        self.engine_config.Basic_Config.num_threads = 16

        # Determine the number of host kv slots.
        self.engine_config.KV_Storage_Config.reserved_length = (
            self.engine_config.Basic_Config.padding_length
            + self.engine_config.Basic_Config.max_decoding_length
        )
        self.engine_config.KV_Storage_Config.slot_byte_size = (
            self.engine_config.KV_Storage_Config.reserved_length
            * self.model_config.compressed_kv_dim
        )
        self.engine_config.KV_Storage_Config.num_host_slots = (
            self.host_kv_cache_byte_size
            // self.engine_config.KV_Storage_Config.slot_byte_size
            // self.model_config.num_hidden_layers
        )
        self.engine_config.KV_Storage_Config.storage_byte_size = (
            self.host_kv_cache_byte_size
        )
        # logging.info(
        #     f"KV storage byte size: {self.engine_config.KV_Storage_Config.storage_byte_size}"
        # )
        logging.info(
            f"Number of host kv slots: {self.engine_config.KV_Storage_Config.num_host_slots}"
        )

        self.engine_config.GPU_Buffer_Config.kv_buffer_num_tokens = (
            self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size
            * (
                self.engine_config.Basic_Config.max_decoding_length
                + self.engine_config.Basic_Config.padding_length
            )
        )

        self.engine_config.GPU_Buffer_Config.module_shapes = {
            "attn": {
                "q_proj.weight": [
                    self.hf_config.attn_config.hidden_size,
                    self.hf_config.attn_config.num_heads
                    * self.hf_config.attn_config.head_dim,
                ],
                "q_norm.weight": [self.hf_config.attn_config.hidden_size],
                "k_proj.weight": [
                    self.hf_config.attn_config.hidden_size,
                    self.hf_config.attn_config.num_key_value_heads
                    * self.hf_config.attn_config.head_dim,
                ],
                "v_proj.weight": [
                    self.hf_config.attn_config.hidden_size,
                    self.hf_config.attn_config.num_key_value_heads
                    * self.hf_config.attn_config.head_dim,
                ],
                "k_norm.weight": [self.hf_config.attn_config.hidden_size],
                "o_proj.weight": [
                    self.hf_config.attn_config.num_heads
                    * self.hf_config.attn_config.head_dim,
                    self.hf_config.attn_config.hidden_size,
                ],
                "post_attention_layernorm.weight": [
                    self.hf_config.attn_config.hidden_size
                ],
                "input_layernorm.weight": [
                    self.hf_config.attn_config.hidden_size
                ],
            },
            "routed_expert": {
                "gate_proj.weight": [
                    self.hf_config.moe_config.hidden_size,
                    self.hf_config.moe_config.intermediate_size,
                ],
                "up_proj.weight": [
                    self.hf_config.moe_config.hidden_size,
                    self.hf_config.moe_config.intermediate_size,
                ],
                "down_proj.weight": [
                    self.hf_config.moe_config.intermediate_size,
                    self.hf_config.moe_config.hidden_size,
                ],
            },
        }

    def _parse_model_config(self):
        model_config = ModelConfig()

        self.hf_config = HuggingFaceModelConfig(self.hf_model_config)

        model_config.model_type = self.hf_config.model_type
        model_config.num_hidden_layers = self.hf_config.num_layers
        model_config.num_local_experts = self.hf_config.moe_config.num_experts
        model_config.num_attention_heads = self.hf_config.attn_config.num_heads
        model_config.num_key_value_heads = (
            self.hf_config.attn_config.num_key_value_heads
        )
        model_config.head_dim = self.hf_config.attn_config.head_dim
        return model_config

    def _load_model_skeleton(self):
        for key, param in self.model.named_parameters():
            if key in self.skeleton_state_dict:
                param.data = self.skeleton_state_dict[key]

        model_skeletion_byte_size = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        ) / (1024**3)
        logging.info(f"Model skeleton size: {model_skeletion_byte_size:.2f} GB")

    def Init(self):
        try:
            # self._init_torch_dist_nccl()
            torch.cuda.set_device(self.local_rank)
            # self._parse_state_dict_ep()
            # logging.info("State dict parsed")
            self.core_engine = core_engine(
                self.engine_config, self.model_config
            )
            logging.info("Core engine created")
            logging.info(f"_name_or_path: {self.hf_model_config._name_or_path}")
            if (
                "deepseek-ai/DeepSeek-V3" in self.hf_model_config._name_or_path
                or "deepseek-ai/DeepSeek-R1"
                in self.hf_model_config._name_or_path
            ):
                param_byte_size = 675 * 1024 * 1024 * 1024
            else:
                raise ValueError("Unknown huggingface model card")

            self.core_engine.Init(
                self.shm_name, self.tensor_meta_shm_name, param_byte_size
            )
            logging.info("Core engine initialized")

            logging.info("Host to Device worker started")
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e
        return (
            self.core_engine,
            self.model,
            self.engine_config,
            self.model_config,
            self.hf_model_config,
        )

    def _parse_state_dict_ep(self):
        model_init_start_time = time.perf_counter()
        self.hf_model_config._attn_implementation = "eager"
        self.model = Qwen3MoeForCausalLM._from_config(self.hf_model_config).to(
            self.engine_config.Basic_Config.device_torch
        )
        self.model.eval()
        logging.info(
            f"torch module init time: {time.perf_counter() - model_init_start_time} s"
        )

        self.weight_copy_task["attn"] = []
        self.weight_copy_task["routed_expert"] = []
        self.weight_copy_task["shared_expert"] = []

        # We have 8 devices and 256 experts per layer.
        # Each devices is responsible for 32 experts. However, the device may not be able to hold all experts.
        # In this case, we hold NUM_LOCAL_EXPERT_PER_LAYER in the GPU and the 32 - NUM_LOCAL_EXPERT_PER_LAYER in the host memory.
        # So the self.local_routed_experts in just the names of experts in each rank's GPU.

        self.local_routed_experts = []
        self.host_routed_experts = []
        # self.expert_location_map = {}

        NUM_LOCAL_EXPERT_PER_LAYER = 24

        routed_expert_gpu_start_idx = self.rank * 32
        routed_expert_gpu_end_idx = (
            routed_expert_gpu_start_idx + NUM_LOCAL_EXPERT_PER_LAYER
        )
        routed_expert_host_start_idx = routed_expert_gpu_end_idx
        routed_expert_host_end_idx = (self.rank + 1) * 32
        for layer_idx in range(
            self.hf_model_config.first_k_dense_replace,
            self.model_config.num_hidden_layers,
        ):
            # We split the 256 into 8 parts, each part is 32 experts.
            # The first NUM_LOCAL_EXPERT_PER_LAYER in each part associated with the corresponding rank.
            # The rest of the experts in the part are stored in the host memory.
            for expert_idx in range(
                routed_expert_gpu_start_idx, routed_expert_gpu_end_idx
            ):
                self.local_routed_experts.append(
                    "routed_expert_" + str(layer_idx) + "_" + str(expert_idx)
                )
            for expert_idx in range(
                routed_expert_host_start_idx, routed_expert_host_end_idx
            ):
                self.host_routed_experts.append(
                    "routed_expert_" + str(layer_idx) + "_" + str(expert_idx)
                )

        self.weight_copy_task["routed_expert"] = self.host_routed_experts
        # assert len(self.weight_copy_task["routed_expert"]) == 0

        for layer_idx in trange(self.model_config.num_hidden_layers):
            for name, _ in self.model.model.layers[
                layer_idx
            ].self_attn.named_parameters():
                tensor_full_name = (
                    "model.layers." + str(layer_idx) + ".self_attn." + name
                )
                self.state_dict_name_map[tensor_full_name] = {
                    "module_key": "attn_" + str(layer_idx),
                    "tensor_key": name,
                }
            self.weight_copy_task["attn"].append("attn_" + str(layer_idx))

            # if layer_idx >= self.hf_model_config.first_k_dense_replace:
            # for name, _ in self.model.model.layers[
            #     layer_idx
            # ].mlp.shared_experts.named_parameters():
            # tensor_full_name = (
            #     "model.layers."
            #     + str(layer_idx)
            #     + ".mlp.shared_experts."
            #     + name
            # )
            # self.state_dict_name_map[tensor_full_name] = {
            #     "module_key": "shared_expert_" + str(layer_idx),
            #     "tensor_key": name,
            # }
            # self.weight_copy_task["shared_expert"].append(
            #     "shared_expert_" + str(layer_idx)
            # )

            for expert_idx in range(self.model_config.num_local_experts):
                for name, _ in (
                    self.model.model.layers[layer_idx]
                    .mlp.experts[expert_idx]
                    .named_parameters()
                ):
                    tensor_full_name = (
                        "model.layers."
                        + str(layer_idx)
                        + ".mlp.experts."
                        + str(expert_idx)
                        + "."
                        + name
                    )
                    self.state_dict_name_map[tensor_full_name] = {
                        "module_key": "routed_expert_"
                        + str(layer_idx)
                        + "_"
                        + str(expert_idx),
                        "tensor_key": name,
                    }
