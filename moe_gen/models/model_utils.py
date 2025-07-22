import logging
import os
from multiprocessing import Process
from typing import Optional

import torch
from deepseek.deepseekv3.configuration_deepseek_v3 import DeepseekV3Config
from qwen.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from safetensors.torch import load_file
from tqdm import tqdm

from ..config.config import EngineConfig, ModelConfig
from ..config.hf_config_parser import HuggingFaceModelConfig

SUPPORTED_MODELS = {
    "deepseek-ai/DeepSeek-V3": DeepseekV3Config,
    "deepseek-ai/DeepSeek-R1": DeepseekV3Config,
    "Qwen/Qwen3-235B-A22B": Qwen3MoeConfig,
    "Qwen/Qwen3-30B-A3B": Qwen3MoeConfig,
}


class ModelInitializer:
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
        self.cache_dir = hf_cache_dir if cache_dir is None else cache_dir
        self.engine_config = engine_config
        self.skeleton_state_dict = skeleton_state_dict
        self.shm_name = shm_name
        self.tensor_meta_shm_name = tensor_meta_shm_name
        self.pt_ckpt_dir = pt_ckpt_dir
        self.host_kv_cache_size = host_kv_cache_size
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size

        self.host_kv_cache_byte_size = host_kv_cache_size * (1024**3)

        assert self.huggingface_ckpt_name in SUPPORTED_MODELS, (
            f"Unsupported model: {self.huggingface_ckpt_name}. "
            f"Supported models are: {list(SUPPORTED_MODELS.keys())}"
        )

        self.hf_model_config = SUPPORTED_MODELS[
            self.huggingface_ckpt_name
        ].from_pretrained(self.cache_dir, trust_remote_code=True)

    def _save_safetensors_to_pt(self):
        ckpt_files = os.listdir(self.cache_dir)
        ckpt_files = [
            os.path.join(self.cache_dir, ckpt)
            for ckpt in ckpt_files
            if ckpt.endswith(".safetensors")
        ]

        def save_and_load(file_path, save_dir):
            tensor_dict = load_file(file_path)
            torch.save(tensor_dict, save_dir)
            return tensor_dict

        processes = []
        for ckpt in tqdm(
            ckpt_files, desc="Loading checkpoint files", smoothing=0
        ):
            dst_dir = os.path.join(
                self.pt_ckpt_dir,
                ckpt.split("/")[-1].replace(".safetensors", ".pt"),
            )
            if os.path.exists(dst_dir):
                continue

            p = Process(target=save_and_load, args=(ckpt, dst_dir))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            p.close()
        logging.info("All safetensor loader processes joined")

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
        model_config.compressed_kv_dim = (
            self.hf_config.attn_config.compressed_kv_dim
            if hasattr(self.hf_config.attn_config, "compressed_kv_dim")
            else None
        )
        return model_config
