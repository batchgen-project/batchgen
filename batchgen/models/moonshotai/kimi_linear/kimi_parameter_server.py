# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""Kimi-Linear Parameter Server for BatchGen.

Checkpoint format (plain HF safetensors, BF16; no language_model. prefix):
    model.layers.{L}.self_attn.{name}                          → MLA layers
    model.layers.{L}.self_attn.{name}                          → KDA layers
    model.layers.{L}.block_sparse_moe.experts.{E}.w{1,2,3}     → routed experts
    model.layers.{L}.block_sparse_moe.shared_experts.{...}     → shared expert

BatchGen module mapping:
    attn_{L}              → NoPE-MLA layers (7)
    kda_attn_{L}          → KDA layers (20)
    routed_expert_{L}_{E} → BF16 routed experts (MoE layers)
    shared_expert_{L}     → BF16 shared expert (MoE layers)

Skeleton (NOT in map; loaded by the PSM from skeleton_state_dict):
    norms, router gate (+ e_score_correction_bias), dense layer-0 MLP,
    embeddings, final norm, lm_head.
"""

import gc
import logging
import os
import shutil
import uuid

import torch
from tqdm import trange

from batchgen.config.model_registry import load_config
from batchgen.ckpt_converter.ckpt_converter import ckpt_converter

try:
    from batchgen.core_engine import Parameter_Server
except ImportError:
    from batchgen.models.engine_loader import core_engine
    Parameter_Server = core_engine.Parameter_Server


_MLA_ATTN_TENSOR_NAMES = [
    "q_proj.weight",
    "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight",
    "kv_b_proj.weight",
    "o_proj.weight",
]

_KDA_ATTN_TENSOR_NAMES = [
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "q_conv1d.weight",
    "k_conv1d.weight",
    "v_conv1d.weight",
    "A_log",
    "f_a_proj.weight",
    "f_b_proj.weight",
    "dt_bias",
    "b_proj.weight",
    "g_a_proj.weight",
    "g_b_proj.weight",
    "o_norm.weight",
    "o_proj.weight",
]

_SHARED_EXPERT_TENSOR_NAMES = [
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
]

_ROUTED_EXPERT_TENSOR_NAMES = [
    "w1.weight",
    "w2.weight",
    "w3.weight",
]


class KimiLinear_Parameter_Server:
    """Parameter server for Kimi-Linear (all weights BF16)."""

    def __init__(self, huggingface_ckpt_name, cache_dir, converted_ckpt_dir,
                 enable_hugetlbfs, enable_memfd=False):
        self.cache_dir = cache_dir
        self.huggingface_ckpt_name = huggingface_ckpt_name
        self.converted_ckpt_dir = converted_ckpt_dir
        self.weight_copy_task = {}
        self.state_dict_name_map = {}
        self.enable_hugetlbfs = enable_hugetlbfs
        self.enable_memfd = enable_memfd

        self.model_config = load_config(huggingface_ckpt_name)
        # Name-pattern lookup drops linear_attn_config; config.json in
        # cache_dir is authoritative for the KDA/MLA layer split.
        if (
            getattr(self.model_config, "linear_attn_config", None) is None
            and cache_dir
        ):
            cfg_json = os.path.join(cache_dir, "config.json")
            if os.path.isfile(cfg_json):
                from .config import KimiLinearConfig

                self.model_config = KimiLinearConfig.from_json(cfg_json)
        self.num_layers = self.model_config.num_hidden_layers  # 27
        self.num_experts = getattr(self.model_config, "n_routed_experts", 256) or 256
        self.first_k_dense_replace = self.model_config.first_k_dense_replace  # 1

        free_memory, total_memory = torch.cuda.mem_get_info()
        logging.info(
            f"Python PM instantiation: GPU 0 free memory: "
            f"{free_memory / 1024**3:.2f} GB / {total_memory / 1024**3:.2f} GB"
        )

    def _is_kda_layer(self, layer_idx: int) -> bool:
        return self.model_config.is_kda_layer(layer_idx)

    def Init(self):
        self._parse_state_dict()

        self.parameter_server = Parameter_Server(self.enable_hugetlbfs, self.enable_memfd)

        # Kimi-Linear: ~96 GiB BF16 total; 110 GiB with buffer.
        byte_size = 110 * 1024 * 1024 * 1024

        total, used, free = shutil.disk_usage("/dev/shm")
        logging.info(f"Freespace in /dev/shm: {free / 1024**3:.2f} GB")
        if free < byte_size:
            raise ValueError(
                f"Shared memory size is not enough. Required: {byte_size}, "
                f"Available: {free}. Please clear /dev/shm or increase it."
            )

        self.shm_name = "/shm_" + str(uuid.uuid4())
        self.tensor_meta_shm_name = "/shm_" + str(uuid.uuid4())
        logging.info(f"Model parameters shared memory name: {self.shm_name}")
        logging.info(f"Byte size: {byte_size}")

        # Plain HF safetensors — convert to BatchGen binary format on first run
        # (cached under <cache_dir>/converted_ckpt afterwards).
        if self.converted_ckpt_dir is None or not os.path.isdir(self.converted_ckpt_dir):
            converter = ckpt_converter()
            self.converted_ckpt_dir = converter.convert_model_directory(
                self.cache_dir, marlin=False
            )
        else:
            logging.info(f"Using pre-converted checkpoint: {self.converted_ckpt_dir}")

        self.parameter_server.Init(
            self.shm_name,
            self.tensor_meta_shm_name,
            byte_size,
            str(self.converted_ckpt_dir),
            self.state_dict_name_map,
        )
        return self.shm_name, self.tensor_meta_shm_name

    def _parse_state_dict(self):
        self.weight_copy_task["attn"] = []
        self.weight_copy_task["kda_attn"] = []
        self.weight_copy_task["routed_expert"] = []
        self.weight_copy_task["shared_expert"] = []

        for layer_idx in trange(self.num_layers, desc="Parsing state_dict"):
            # --- Attention: MLA vs KDA by layer type ---
            if self._is_kda_layer(layer_idx):
                for name in _KDA_ATTN_TENSOR_NAMES:
                    full = f"model.layers.{layer_idx}.self_attn.{name}"
                    self.state_dict_name_map[full] = {
                        "module_key": f"kda_attn_{layer_idx}",
                        "tensor_key": name,
                    }
                self.weight_copy_task["kda_attn"].append(f"kda_attn_{layer_idx}")
            else:
                for name in _MLA_ATTN_TENSOR_NAMES:
                    full = f"model.layers.{layer_idx}.self_attn.{name}"
                    self.state_dict_name_map[full] = {
                        "module_key": f"attn_{layer_idx}",
                        "tensor_key": name,
                    }
                self.weight_copy_task["attn"].append(f"attn_{layer_idx}")

            # --- MoE (layer 0 is dense) ---
            if layer_idx >= self.first_k_dense_replace:
                for name in _SHARED_EXPERT_TENSOR_NAMES:
                    full = (
                        f"model.layers.{layer_idx}.block_sparse_moe."
                        f"shared_experts.{name}"
                    )
                    self.state_dict_name_map[full] = {
                        "module_key": f"shared_expert_{layer_idx}",
                        "tensor_key": name,
                    }
                self.weight_copy_task["shared_expert"].append(
                    f"shared_expert_{layer_idx}"
                )

                for expert_idx in range(self.num_experts):
                    for name in _ROUTED_EXPERT_TENSOR_NAMES:
                        full = (
                            f"model.layers.{layer_idx}.block_sparse_moe."
                            f"experts.{expert_idx}.{name}"
                        )
                        self.state_dict_name_map[full] = {
                            "module_key": f"routed_expert_{layer_idx}_{expert_idx}",
                            "tensor_key": name,
                        }
                    self.weight_copy_task["routed_expert"].append(
                        f"routed_expert_{layer_idx}_{expert_idx}"
                    )

        logging.info(
            f"state_dict_name_map: {len(self.state_dict_name_map)} entries "
            f"(kda_attn={len(self.weight_copy_task['kda_attn'])} layers, "
            f"attn={len(self.weight_copy_task['attn'])} layers, "
            f"routed_expert={len(self.weight_copy_task['routed_expert'])} modules)"
        )
        gc.collect()
        torch.cuda.empty_cache()
