# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
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
from multiprocessing import Process
from typing import Optional, Union, List

import torch
import torch.nn as nn
import triton.language as tl
import torch.distributed as dist

# from batchgen.config import EngineConfig, ModelConfig
from ....config.config import EngineConfig, ModelConfig
from .configuration_deepseek_v3 import DeepseekV3Config
try:
    from batchgen.core_engine import batchgen as core_engine
except ImportError:
    # jit compile
    from core_engine import batchgen as core_engine

from typing import Tuple
from ....config.engine_config_parser import parse_config_from_json
from .set_basic_config import set_basic_config

def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
        x: the dividend.
        y: the divisor.

    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y

def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(
        torch.float8_e4m3fn
    ).view(m, n), (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128),
        dtype=x.dtype,
        device=x.device,
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (
        x_amax / 448.0
    ).view(x_view.size(0), x_view.size(2))

def compressed_kv_fp8_to_bf16_per_token(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    Dequantize the output of bf16_to_fp8_per_token back to BF16.
    Inputs:
      q     [bsz, seq, 576]  dtype=float8_e4m3fn
      scale [bsz, seq]       dtype=float32
    Returns:
      x_bf16 [bsz, seq, 576] dtype=bfloat16
    """
    bsz, seq_len, dim = q.shape
    M = bsz * seq_len
    # flatten
    q_flat = q.view(M, dim).float()               # upcast FP8→FP32
    x_rec = q_flat * s.view(M, 1)             # rescale
    return x_rec.to(torch.bfloat16).view(bsz, seq_len, dim)	

def deepseek_v3_dequantization(
    weight_data_fp8: torch.Tensor,
    weight_scale_inv_fp32: torch.Tensor,
    block_size=(128, 128),
) -> torch.Tensor:
    """
    Vectorized dequantization that removes Python-level loops
    and leverages PyTorch's parallelism.
    """
    rows, cols = weight_data_fp8.shape
    block_rows, block_cols = block_size

    # Number of blocks in each dimension
    n_block_rows = rows // block_rows
    n_block_cols = cols // block_cols

    # 1) Reshape weight data into 4D block form and cast to float32
    #    shape becomes [n_block_rows, block_rows, n_block_cols, block_cols].
    weight_4d = weight_data_fp8.reshape(
        n_block_rows, block_rows, n_block_cols, block_cols
    ).to(torch.float32)

    # 2) Broadcast scale into 4D by unsqueezing along the second and fourth dimensions.
    #    shape becomes [n_block_rows, 1, n_block_cols, 1].
    scale_4d = weight_scale_inv_fp32.unsqueeze(1).unsqueeze(-1)

    # 3) Multiply once using broadcasting
    dequantized_4d = weight_4d * scale_4d

    # 4) Reshape back to [rows, cols] and cast to bfloat16
    dequantized_weight = dequantized_4d.reshape(rows, cols).to(torch.bfloat16)

    return dequantized_weight



class DeepSeekV3_Initializer:
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
        self.hf_model_config = DeepseekV3Config()
        self.hf_model_config._name_or_path = huggingface_ckpt_name
        self.hf_model_config.architectures = ["DeepseekV3ForCausalLM"]

        self.host_kv_cache_size = host_kv_cache_size
        self.host_kv_cache_byte_size = host_kv_cache_size * (1024**3)

        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size    
        self.enable_hugetlbfs = os.environ.get("BATCHGEN_ENABLE_HUGETLBFS", "0") == "1"
        logging.info(f"Enable hugetlbfs: {self.enable_hugetlbfs}")

        # TODO:
        self.model_config = self._parse_model_config()
        self._default_engine_config()
        # self.engine_config = set_basic_config(self.engine_config)

        self.shm_name = shm_name
        self.tensor_meta_shm_name = tensor_meta_shm_name

    def _default_engine_config(self):
        props = torch.cuda.get_device_properties(
            self.engine_config.Basic_Config.device
        )
        total_memory = props.total_memory / (1024**3)
        logging.info(f"Current device total memory: {total_memory} GB")



        # Determine the number of host kv slots.
        self.engine_config.KV_Storage_Config.reserved_length = (
            self.engine_config.Basic_Config.padding_length
            + self.engine_config.Basic_Config.max_decoding_length
        )
        self.engine_config.KV_Storage_Config.slot_byte_size = (
            self.engine_config.KV_Storage_Config.reserved_length
            * self.model_config.compressed_kv_dim
        )  # KV saved in FP8
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
                "q_a_proj.weight": [1536, 7168],
                "q_a_layernorm.weight": [1536],
                "q_b_proj.weight": [24576, 1536],
                "kv_a_proj_with_mqa.weight": [576, 7168],
                "kv_a_layernorm.weight": [512],
                "kv_b_proj.weight": [32768, 512],
                "o_proj.weight": [7168, 16384],
            },
            "routed_expert": {
                "gate_proj.weight": [2048, 7168],
                "up_proj.weight": [2048, 7168],
                "down_proj.weight": [7168, 2048],
            },
            "shared_expert": {
                "gate_proj.weight": [2048, 7168],
                "up_proj.weight": [2048, 7168],
                "down_proj.weight": [7168, 2048],
            },
        }

    def _parse_model_config(self):
        model_config = ModelConfig()

        model_config.model_type = "deepseek_v3"
        model_config.num_hidden_layers = 61
        model_config.num_local_experts = 256
        model_config.num_attention_heads = 128
        model_config.num_key_value_heads = 128
        model_config.head_dim = 192
        model_config.compressed_kv_dim = 576
        return model_config

    def Init(self):
        try:
            torch.cuda.set_device(self.local_rank)
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
                self.shm_name,
                self.tensor_meta_shm_name,
                param_byte_size,
                self.enable_hugetlbfs
            )
            logging.info("Core engine initialized")
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e
        return (
            self.core_engine,
            self.model,
            self.engine_config,
            self.model_config,
            self.hf_model_config
        )


        
        
                