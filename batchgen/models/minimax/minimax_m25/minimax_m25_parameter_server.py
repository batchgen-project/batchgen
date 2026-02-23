# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
# ---------------------------------------------------------------------------- #

"""MiniMax-M2.5 Parameter Server for BatchGen.

Handles weight loading for MiniMax-M2.5 (230B MoE) with:
- FP8 e4m3fn quantized expert weights (block_size [128,128])
- FP8 e4m3fn quantized attention weights (Q/K/V/O projections) with F32 scales
- BF16 QK norm, embeddings, layer norms
- F32 router gate weight, e_score_correction_bias
- 256 routed experts per MoE layer, no shared experts
- All 62 layers are MoE

Checkpoint format (HuggingFace FP8):
    model.layers.{L}.self_attn.{q,k,v,o}_proj.weight            -> FP8 e4m3fn
    model.layers.{L}.self_attn.{q,k,v,o}_proj.weight_scale_inv  -> F32
    model.layers.{L}.self_attn.{q,k}_norm.weight                -> BF16 (QK norm)
    model.layers.{L}.block_sparse_moe.gate.weight                -> F32 (router)
    model.layers.{L}.block_sparse_moe.e_score_correction_bias   -> F32 (routing bias)
    model.layers.{L}.block_sparse_moe.experts.{E}.{w1,w2,w3}.weight           -> FP8 e4m3fn
    model.layers.{L}.block_sparse_moe.experts.{E}.{w1,w2,w3}.weight_scale_inv -> F32

BatchGen format:
    attn_{L}              / {tensor_key}  -> attention FP8 weights + F32 scales + BF16 norms
    routed_expert_{L}_{E} / {tensor_key}  -> expert FP8 weights + F32 scales
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


# GQA attention parameter names (FP8 weights + F32 scales + BF16 norms)
_GQA_ATTN_TENSOR_NAMES = [
    "q_proj.weight",
    "q_proj.weight_scale_inv",
    "k_proj.weight",
    "k_proj.weight_scale_inv",
    "v_proj.weight",
    "v_proj.weight_scale_inv",
    "o_proj.weight",
    "o_proj.weight_scale_inv",
    "q_norm.weight",
    "k_norm.weight",
]

# FP8 expert parameter names
_FP8_EXPERT_TENSOR_NAMES = [
    "w1.weight",          # gate_proj (FP8)
    "w1.weight_scale_inv",
    "w2.weight",          # down_proj (FP8)
    "w2.weight_scale_inv",
    "w3.weight",          # up_proj (FP8)
    "w3.weight_scale_inv",
]


class MiniMaxM25_Parameter_Server:
    """Parameter server for MiniMax-M2.5 with FP8 weight handling."""

    def __init__(self, huggingface_ckpt_name, cache_dir, converted_ckpt_dir, enable_hugetlbfs):
        self.cache_dir = cache_dir
        self.huggingface_ckpt_name = huggingface_ckpt_name
        self.converted_ckpt_dir = converted_ckpt_dir
        self.weight_copy_task = {}
        self.state_dict_name_map = {}
        self.enable_hugetlbfs = enable_hugetlbfs

        self.model_config = load_config(huggingface_ckpt_name)
        self.num_layers = self.model_config.num_hidden_layers        # 62
        self.num_experts = self.model_config.num_local_experts       # 256

        free_memory, total_memory = torch.cuda.mem_get_info()
        gpu0_memory = free_memory / 1024 / 1024 / 1024
        total_memory_gb = total_memory / 1024 / 1024 / 1024
        logging.info(f"Python PM instantiation: GPU 0 free memory: {gpu0_memory:.2f} GB / {total_memory_gb:.2f} GB")

    def Init(self):
        """Initialize parameter server and load weights."""
        self._parse_state_dict()

        self.parameter_server = Parameter_Server(self.enable_hugetlbfs)

        # M2.5: ~225GB FP8 experts + ~8GB BF16 (attn/embed/router) ≈ 233GB, 250GB with buffer
        byte_size = 250 * 1024 * 1024 * 1024

        total, used, free = shutil.disk_usage("/dev/shm")
        logging.info(f"Freespace in /dev/shm: {free/1024/1024/1024:.2f} GB")
        if free < byte_size:
            raise ValueError(
                f"Shared memory size is not enough. Required: {byte_size}, Available: {free}. "
                "Please clear /dev/shm or increase the size."
            )

        self.shm_name = "/shm_" + str(uuid.uuid4())
        self.tensor_meta_shm_name = "/shm_" + str(uuid.uuid4())
        logging.info(f"Model parameters shared memory name: {self.shm_name}")
        logging.info(f"Tensor meta shared memory name: {self.tensor_meta_shm_name}")

        converter = ckpt_converter()
        self.converted_ckpt_dir = converter.convert_model_directory(self.cache_dir)

        self.parameter_server.Init(
            self.shm_name,
            self.tensor_meta_shm_name,
            byte_size,
            self.converted_ckpt_dir,
            self.state_dict_name_map,
        )

        # Expose weights_storage for core engine
        self._weights_storage = self.parameter_server

        return self.shm_name, self.tensor_meta_shm_name

    @property
    def weights_storage(self):
        """Return the C++ parameter server as weights storage for core engine."""
        return self._weights_storage

    def _parse_state_dict(self):
        """Build name mapping from checkpoint tensor names to BatchGen format.

        NOTE: Tensor names need verification against actual checkpoint.
        The mapping below uses the expected HuggingFace naming convention.
        """
        self.weight_copy_task["attn"] = []
        self.weight_copy_task["routed_expert"] = []

        for layer_idx in trange(self.num_layers, desc="Parsing state_dict"):
            # Attention weights (BF16 GQA)
            for name in _GQA_ATTN_TENSOR_NAMES:
                tensor_full_name = f"model.layers.{layer_idx}.self_attn.{name}"
                self.state_dict_name_map[tensor_full_name] = {
                    "module_key": f"attn_{layer_idx}",
                    "tensor_key": name,
                }
            self.weight_copy_task["attn"].append(f"attn_{layer_idx}")

            # All layers are MoE (no first_k_dense_replace)
            for expert_idx in range(self.num_experts):
                for name in _FP8_EXPERT_TENSOR_NAMES:
                    tensor_full_name = (
                        f"model.layers.{layer_idx}.block_sparse_moe.experts."
                        f"{expert_idx}.{name}"
                    )
                    self.state_dict_name_map[tensor_full_name] = {
                        "module_key": f"routed_expert_{layer_idx}_{expert_idx}",
                        "tensor_key": name,
                    }
                self.weight_copy_task["routed_expert"].append(
                    f"routed_expert_{layer_idx}_{expert_idx}"
                )

        attn_count = sum(1 for k in self.state_dict_name_map if "self_attn" in k)
        expert_count = sum(1 for k in self.state_dict_name_map if "experts" in k)
        logging.info(
            f"state_dict_name_map: {len(self.state_dict_name_map)} entries "
            f"(attn={attn_count}, routed_expert={expert_count})"
        )

        gc.collect()
        torch.cuda.empty_cache()
