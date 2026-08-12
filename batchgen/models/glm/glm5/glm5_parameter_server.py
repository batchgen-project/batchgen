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
- byte_size: ~760 GB (FP8 experts) or ~1400 GB (BF16 experts)
"""

import sys as _diag_sys
import time as _diag_time
def _diag(msg):
    print(f"[DIAG {_diag_time.time():.3f}] glm5_ps_import: {msg}", flush=True)
    _diag_sys.stdout.flush()

_diag("start")
import gc
import logging
import os
import shutil
from multiprocessing import Process
_diag("stdlib done")

import torch
_diag("torch done")
from safetensors.torch import load_file
_diag("safetensors done")
from tqdm import tqdm, trange
_diag("tqdm done")

from .model import Glm5ForCausalLM
_diag("model (Glm5ForCausalLM) done")
from batchgen.config.batchgen_model_config import BatchGenModelConfig
_diag("model_registry done")

try:
    from batchgen.core_engine import Parameter_Server
    _diag("core_engine (prebuilt) done")
except ImportError:
    _diag("core_engine ImportError -> engine_loader JIT")
    from batchgen.models.engine_loader import core_engine
    Parameter_Server = core_engine.Parameter_Server
    _diag("engine_loader done")


class GLM5_Parameter_Server:
    def __init__(self, huggingface_ckpt_name, cache_dir, converted_ckpt_dir, enable_hugetlbfs, enable_memfd=False):
        self.cache_dir = cache_dir
        self.huggingface_ckpt_name = huggingface_ckpt_name
        self.converted_ckpt_dir = converted_ckpt_dir
        self.weight_copy_task = {}
        self.state_dict_name_map = {}
        self.enable_hugetlbfs = enable_hugetlbfs
        self.enable_memfd = enable_memfd
        # Single resolved internal config (checkpoint-backed GLM5Config/GLM52Config).
        # Used both for metadata reads and to build the model graph — GLM-5 no
        # longer uses an HF transformers.PretrainedConfig.
        self.model_config = BatchGenModelConfig.resolve(huggingface_ckpt_name, cache_dir)
        self.hf_config = self.model_config
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

        self.parameter_server = Parameter_Server(self.enable_hugetlbfs, self.enable_memfd)

        # GLM-5-FP8: FP8 experts (~675 GB) + FP8 attn + rest ≈ 700 GB
        # GLM-5: BF16 experts (~1350 GB) + FP8 attn + rest ≈ 1380 GB
        if "fp8" in self.huggingface_ckpt_name.lower():
            byte_size = 760 * 1024 * 1024 * 1024
        else:
            byte_size = 1400 * 1024 * 1024 * 1024

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
            # Attention parameters (EXCLUDING indexer — indexer goes to skeleton;
            # EXCLUDING q_a_layernorm/kv_a_layernorm — tiny BF16 RMSNorm weights,
            # route them through skeleton too. If they stay in state_dict_name_map
            # they get stripped from the skeleton loader AND never actually
            # written into the live module at prefill time, so the module keeps
            # its `ones_()` init and every attention layer's Q/K norm is wrong.
            # q_a_layernorm is [2048] bf16 = 4KB/layer, kv_a_layernorm is [512]
            # bf16 = 1KB/layer → ~400KB total, trivial to hold in skeleton.)
            for name, _ in model.model.layers[layer_idx].self_attn.named_parameters():
                if name.startswith("indexer."):
                    continue
                if name in ("q_a_layernorm.weight", "kv_a_layernorm.weight"):
                    continue
                tensor_full_name = f"model.layers.{layer_idx}.self_attn.{name}"
                self.state_dict_name_map[tensor_full_name] = {
                    "module_key": f"attn_{layer_idx}",
                    "tensor_key": name,
                }
            self.weight_copy_task["attn"].append(f"attn_{layer_idx}")

            # MoE layers (layer_idx >= first_k_dense)
            if layer_idx >= first_k_dense:
                # Shared experts — use static param names (experts are placeholders)
                _shared_expert_param_names = ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
                for name in _shared_expert_param_names:
                    tensor_full_name = f"model.layers.{layer_idx}.mlp.shared_experts.{name}"
                    self.state_dict_name_map[tensor_full_name] = {
                        "module_key": f"shared_expert_{layer_idx}",
                        "tensor_key": name,
                    }
                self.weight_copy_task["shared_expert"].append(
                    f"shared_expert_{layer_idx}"
                )

                # Routed experts — use static param names (experts are placeholders)
                _expert_param_names = ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
                num_experts = self.model_config.num_local_experts  # 256
                for expert_idx in range(num_experts):
                    module_key = f"routed_expert_{layer_idx}_{expert_idx}"
                    for name in _expert_param_names:
                        tensor_full_name = (
                            f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{name}"
                        )
                        self.state_dict_name_map[tensor_full_name] = {
                            "module_key": module_key,
                            "tensor_key": name,
                        }
                    self.weight_copy_task["routed_expert"].append(module_key)

        del model
        gc.collect()
        torch.cuda.empty_cache()
