# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
# ---------------------------------------------------------------------------- #

"""GLM-5 parameter server for BatchGen.

Handles checkpoint loading and tensor name mapping for zai-org/GLM-5-FP8.
Standalone — no cross-model imports.

Key differences from DeepSeek:
- 78 layers (vs 61), 256 experts, first_k_dense_replace=3
- kv_a_proj_with_mqa naming (same as DeepSeek)
- Indexer tensors (wk, wq_b, k_norm, weights_proj) — kept in skeleton (BF16/FP8 mixed)
- e_score_correction_bias in MoE gate
- MTP layer at index 78 (eh_proj, enorm, hnorm, shared_head.norm)
- FP8 byte_size ~760 GB
"""

import gc
import logging
import os
import shutil
from multiprocessing import Process

import torch
from safetensors.torch import load_file
from tqdm import tqdm, trange

from .configuration_glm5 import Glm5Config
from .model import Glm5ForCausalLM
from batchgen.config.model_registry import load_config

try:
    from batchgen.core_engine import Parameter_Server
except ImportError:
    from batchgen.models.engine_loader import core_engine
    Parameter_Server = core_engine.Parameter_Server


class GLM5_Parameter_Server:
    def __init__(self, huggingface_ckpt_name, cache_dir, converted_ckpt_dir, enable_hugetlbfs):
        self.cache_dir = cache_dir
        self.huggingface_ckpt_name = huggingface_ckpt_name
        self.converted_ckpt_dir = converted_ckpt_dir
        self.weight_copy_task = {}
        self.state_dict_name_map = {}
        self.enable_hugetlbfs = enable_hugetlbfs
        self.model_config = load_config(huggingface_ckpt_name)
        self.hf_config = Glm5Config()
        self.hf_config._name_or_path = huggingface_ckpt_name

        free_memory, total_memory = torch.cuda.mem_get_info()
        logging.info(
            f"Python PM instantiation: GPU 0 free memory: "
            f"{free_memory / 1024**3:.1f} GB / {total_memory / 1024**3:.1f} GB"
        )

    def Init(self):
        free_memory, total_memory = torch.cuda.mem_get_info()
        logging.info(
            f"GPU 0 free mem pm start Init: {free_memory / 1024**3:.1f} GB / "
            f"{total_memory / 1024**3:.1f} GB"
        )

        self._parse_state_dict()

        self.parameter_server = Parameter_Server(self.enable_hugetlbfs)

        # GLM-5-FP8: ~756 GB checkpoint
        byte_size = 760 * 1024 * 1024 * 1024

        total, used, free = shutil.disk_usage("/dev/shm")
        logging.info(f"Freespace in /dev/shm: {free / 1024**3:.1f} GB")
        if free < byte_size:
            raise ValueError(
                f"Shared memory size not enough. Required: {byte_size / 1024**3:.0f} GB, "
                f"Available: {free / 1024**3:.0f} GB. "
                f"Please clear /dev/shm or increase size."
            )

        import uuid
        self.shm_name = "/shm_" + str(uuid.uuid4())
        self.tensor_meta_shm_name = "/shm_" + str(uuid.uuid4())
        logging.info(f"Model parameters shared memory name: {self.shm_name}")
        logging.info(f"Tensor meta shared memory name: {self.tensor_meta_shm_name}")
        logging.info(f"Byte size: {byte_size}")

        # Convert checkpoint files
        from batchgen.ckpt_converter.ckpt_converter import ckpt_converter
        converter = ckpt_converter()
        self.converted_ckpt_dir = converter.convert_model_directory(self.cache_dir)

        self.parameter_server.Init(
            self.shm_name,
            self.tensor_meta_shm_name,
            byte_size,
            self.converted_ckpt_dir,
            self.state_dict_name_map,
        )
        return self.shm_name, self.tensor_meta_shm_name

    def _parse_state_dict(self):
        """Build state_dict_name_map by walking model named_parameters.

        GLM-5 differences from DeepSeek:
        - Indexer tensors (self_attn.indexer.*) are EXCLUDED from the map
          (they go to skeleton). Indexer has wk, wq_b (FP8), k_norm, weights_proj.
        - Dense MLP layers (0-2) have mlp.gate_proj/up_proj/down_proj (no experts)
        - MoE layers (3-77) have mlp.gate, mlp.shared_experts, mlp.experts
        - MTP layer 78 has eh_proj, enorm, hnorm, shared_head.norm
        """
        self.hf_config._attn_implementation = "eager"
        model = Glm5ForCausalLM(self.hf_config).to('cpu')
        model.eval()

        self.weight_copy_task["attn"] = []
        self.weight_copy_task["routed_expert"] = []
        self.weight_copy_task["shared_expert"] = []

        num_layers = self.model_config.num_hidden_layers  # 78
        first_k_dense = self.model_config.first_k_dense_replace  # 3

        for layer_idx in trange(num_layers, desc="Parsing state_dict"):
            # Attention parameters (EXCLUDING indexer — indexer goes to skeleton)
            for name, _ in model.model.layers[layer_idx].self_attn.named_parameters():
                # Skip indexer parameters — they stay in skeleton
                if name.startswith("indexer."):
                    continue
                tensor_full_name = f"model.layers.{layer_idx}.self_attn.{name}"
                self.state_dict_name_map[tensor_full_name] = {
                    "module_key": f"attn_{layer_idx}",
                    "tensor_key": name,
                }
            self.weight_copy_task["attn"].append(f"attn_{layer_idx}")

            # MoE layers (layer_idx >= first_k_dense)
            if layer_idx >= first_k_dense:
                # Shared experts
                for name, _ in model.model.layers[
                    layer_idx
                ].mlp.shared_experts.named_parameters():
                    tensor_full_name = f"model.layers.{layer_idx}.mlp.shared_experts.{name}"
                    self.state_dict_name_map[tensor_full_name] = {
                        "module_key": f"shared_expert_{layer_idx}",
                        "tensor_key": name,
                    }
                self.weight_copy_task["shared_expert"].append(
                    f"shared_expert_{layer_idx}"
                )

                # Routed experts
                num_experts = self.model_config.num_local_experts  # 256
                for expert_idx in range(num_experts):
                    for name, _ in (
                        model.model.layers[layer_idx]
                        .mlp.experts[expert_idx]
                        .named_parameters()
                    ):
                        tensor_full_name = (
                            f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{name}"
                        )
                        self.state_dict_name_map[tensor_full_name] = {
                            "module_key": f"routed_expert_{layer_idx}_{expert_idx}",
                            "tensor_key": name,
                        }
                    self.weight_copy_task["routed_expert"].append(
                        f"routed_expert_{layer_idx}_{expert_idx}"
                    )

        del model
        gc.collect()
        torch.cuda.empty_cache()
