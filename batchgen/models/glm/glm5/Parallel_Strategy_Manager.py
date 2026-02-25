# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
# ---------------------------------------------------------------------------- #

"""GLM-5 Parallel Strategy Manager for BatchGen.

Standalone — no cross-model imports.

Key differences from DeepSeek V3 PSM:
- 78 layers (vs 61), hidden_size=6144 (vs 7168)
- Indexer params (self_attn.indexer.*) excluded from state_dict_name_map → skeleton
- kv_a_proj_with_mqa naming (same as DeepSeek)
- DSA dual KV cache handled automatically by batchgen_worker.py
- MoE gate has e_score_correction_bias
- Dense MLP layers 0-2, MoE layers 3-77
- Prefill: all modules offloaded (DP), Decode: EP
"""

import gc
import logging
import os
import time
import types

import torch
import torch.distributed as dist

from .model import Glm5ForCausalLM, Glm5MoEDecode
from .wrappers import GLM5ExpertWrapper, GLM5AttnWrapper




class GLM5ParallelStrategyManager:
    NUM_TOTAL_EXPERTS = 256
    NUM_LAYERS = 78
    FIRST_K_DENSE = 3
    HIDDEN_SIZE = 6144

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
        self.weight_copy_task = {}
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.rank = global_rank
        # Detect FP8 variant by checking for expert scale tensors in skeleton
        self.is_fp8_experts = any(
            "experts.0.gate_proj.weight_scale_inv" in k for k in skeleton_state_dict
        )

    def configure_prefill(self):
        """Configure model for prefill (pure DP, all modules offloaded)."""
        start_time = time.perf_counter()
        timings = {}

        self.loaded_model_config.phase = "prefill"

        step_start = time.perf_counter()
        self.model = Glm5ForCausalLM(self.loaded_model_config)
        timings['model_init'] = time.perf_counter() - step_start

        self.state_dict_name_map = {}
        self.weight_copy_task = {
            "attn": [],
            "routed_expert": [],
            "shared_expert": [],
        }

        step_start = time.perf_counter()
        for layer_idx in range(self.model_config.num_hidden_layers):
            # Attention (EXCLUDING indexer → skeleton)
            for name, _ in self.model.model.layers[layer_idx].self_attn.named_parameters():
                if name.startswith("indexer."):
                    continue
                tensor_full_name = f"model.layers.{layer_idx}.self_attn.{name}"
                self.state_dict_name_map[tensor_full_name] = {
                    "module_key": f"attn_{layer_idx}",
                    "tensor_key": name,
                }
            self.weight_copy_task["attn"].append(f"attn_{layer_idx}")

            if layer_idx >= self.FIRST_K_DENSE:
                # Shared experts — use static param names (no nn.Module traversal)
                _shared_expert_param_names = ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
                for name in _shared_expert_param_names:
                    tensor_full_name = f"model.layers.{layer_idx}.mlp.shared_experts.{name}"
                    self.state_dict_name_map[tensor_full_name] = {
                        "module_key": f"shared_expert_{layer_idx}",
                        "tensor_key": name,
                    }
                self.weight_copy_task["shared_expert"].append(f"shared_expert_{layer_idx}")

                # Routed experts — use static param names (experts are placeholders)
                _expert_param_names = ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
                for expert_idx in range(self.NUM_TOTAL_EXPERTS):
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
        timings['weight_mappings'] = time.perf_counter() - step_start

        step_start = time.perf_counter()
        self._extract_dequantize_scale()
        timings['dequantize'] = time.perf_counter() - step_start

        step_start = time.perf_counter()
        self._load_model_skeleton()
        timings['skeleton'] = time.perf_counter() - step_start

        step_start = time.perf_counter()
        self._config_attn_module()
        timings['attn'] = time.perf_counter() - step_start

        step_start = time.perf_counter()
        self._config_expert_module()
        timings['expert'] = time.perf_counter() - step_start

        self._config_lm_head_hook()
        self.model.eval()

        step_start = time.perf_counter()
        self.model.to(self.engine_config.Basic_Config.device_torch)
        self._setup_fp8_scales()
        timings['to_device'] = time.perf_counter() - step_start

        total_time = time.perf_counter() - start_time
        if self.rank == 0:
            logging.info(
                f"[PREFILL] Model configured in {total_time:.2f}s "
                f"(init={timings['model_init']:.1f}s, skeleton={timings['skeleton']:.1f}s, "
                f"expert={timings['expert']:.1f}s, to_device={timings['to_device']:.1f}s)"
            )
        return self.model, self.weight_copy_task

    def configure_decoding(self, padding_bsz=None, comm=None):
        """Configure model for decode: DP + EP."""
        self.loaded_model_config.phase = "decode"
        self.loaded_model_config._attn_implementation = "eager"
        self.model = None
        torch.cuda.empty_cache()

        self.model = Glm5ForCausalLM(self.loaded_model_config)

        self.weight_copy_task = {
            "attn": [],
            "routed_expert": [],
            "shared_expert": [],
        }
        self.state_dict_name_map = {}
        self.local_routed_experts = []
        self.host_routed_experts = []

        NUM_EXPERT_PER_RANK = self.NUM_TOTAL_EXPERTS // self.world_size

        if self.world_size > 8:
            self.enable_ep_offloading = False
            NUM_LOCAL_EXPERT_PER_LAYER = NUM_EXPERT_PER_RANK
        elif self.engine_config.EP_Config.enable_offloading:
            self.enable_ep_offloading = True
            offload_ratio = self.engine_config.EP_Config.offloading_ratio
            NUM_LOCAL_EXPERT_PER_LAYER = int(NUM_EXPERT_PER_RANK * (1 - offload_ratio))
        else:
            self.enable_ep_offloading = False
            NUM_LOCAL_EXPERT_PER_LAYER = self.engine_config.EP_Config.num_local_expert_per_layer
            if NUM_LOCAL_EXPERT_PER_LAYER is None or NUM_LOCAL_EXPERT_PER_LAYER == 0:
                NUM_LOCAL_EXPERT_PER_LAYER = NUM_EXPERT_PER_RANK

        self.num_local_expert_per_layer = NUM_LOCAL_EXPERT_PER_LAYER

        routed_expert_gpu_start_idx = self.global_rank * NUM_EXPERT_PER_RANK
        routed_expert_gpu_end_idx = routed_expert_gpu_start_idx + NUM_LOCAL_EXPERT_PER_LAYER
        routed_expert_host_end_idx = (self.global_rank + 1) * NUM_EXPERT_PER_RANK

        for layer_idx in range(self.FIRST_K_DENSE, self.model_config.num_hidden_layers):
            for expert_idx in range(routed_expert_gpu_start_idx, routed_expert_gpu_end_idx):
                self.local_routed_experts.append(f"routed_expert_{layer_idx}_{expert_idx}")
            for expert_idx in range(routed_expert_gpu_end_idx, routed_expert_host_end_idx):
                self.host_routed_experts.append(f"routed_expert_{layer_idx}_{expert_idx}")

        self.weight_copy_task["routed_expert"] = self.host_routed_experts

        # Build state_dict_name_map for all modules
        for layer_idx in range(self.model_config.num_hidden_layers):
            for name, _ in self.model.model.layers[layer_idx].self_attn.named_parameters():
                if name.startswith("indexer."):
                    continue
                tensor_full_name = f"model.layers.{layer_idx}.self_attn.{name}"
                self.state_dict_name_map[tensor_full_name] = {
                    "module_key": f"attn_{layer_idx}",
                    "tensor_key": name,
                }

            if layer_idx >= self.FIRST_K_DENSE:
                _shared_expert_param_names = ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
                for name in _shared_expert_param_names:
                    tensor_full_name = f"model.layers.{layer_idx}.mlp.shared_experts.{name}"
                    self.state_dict_name_map[tensor_full_name] = {
                        "module_key": f"shared_expert_{layer_idx}",
                        "tensor_key": name,
                    }

                _expert_param_names = ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
                for expert_idx in range(self.model_config.num_local_experts):
                    module_key = f"routed_expert_{layer_idx}_{expert_idx}"
                    for name in _expert_param_names:
                        tensor_full_name = (
                            f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{name}"
                        )
                        self.state_dict_name_map[tensor_full_name] = {
                            "module_key": module_key,
                            "tensor_key": name,
                        }

        torch.cuda.empty_cache()
        self._extract_dequantize_scale()
        self._load_model_skeleton()
        self._load_local_routed_experts()
        self._load_attn_module()
        self._load_shared_expert_module()
        self._config_attn_module()
        self._config_expert_module()
        self._setup_decode_moe(comm)
        self._config_lm_head_hook()

        self.model.eval()
        self.model.to(self.engine_config.Basic_Config.device_torch)
        self._setup_fp8_scales()

        if self.rank == 0:
            used = torch.cuda.memory_allocated(self.engine_config.Basic_Config.device_torch)
            logging.info(f"[MODEL] GPU memory after init: {used / (1024**3):.2f} GB used")

        self._init_mode_decoding()
        effective_bsz = padding_bsz if padding_bsz is not None else 128
        self._init_decoding_padding_bsz(effective_bsz)

        if os.getenv("BATCHGEN_ENABLE_ALL_TO_ALL", "0") == "1":
            self._init_ata_comms(effective_bsz)

        return self.model, self.weight_copy_task

    def set_num_tokens_per_rank(self, num_tokens_per_rank: int):
        for layer_idx in range(self.FIRST_K_DENSE, self.model_config.num_hidden_layers):
            layer = self.model.model.layers[layer_idx].mlp
            if hasattr(layer, "set_num_tokens_per_rank"):
                layer.set_num_tokens_per_rank(num_tokens_per_rank)

    def _init_decoding_padding_bsz(self, padding_bsz):
        env_max_bsz = os.getenv("BATCHGEN_MAX_RANK_BSZ")
        max_rank_bsz = int(env_max_bsz) if env_max_bsz else padding_bsz
        if self.rank == 0:
            logging.info(f"[DECODE] Padding batch size: {max_rank_bsz}")

        for layer_idx in range(self.FIRST_K_DENSE, self.model_config.num_hidden_layers):
            layer = self.model.model.layers[layer_idx].mlp
            if hasattr(layer, "init_num_tokens"):
                layer.init_num_tokens(max_rank_bsz)

    def _init_mode_decoding(self):
        has_persistent = self.num_local_expert_per_layer > 0
        if not has_persistent:
            if self.rank == 0:
                logging.info("EP offloading: no persistent experts, skipping grouped GEMM init")
            return
        for layer_idx in range(self.FIRST_K_DENSE, self.model_config.num_hidden_layers):
            layer = self.model.model.layers[layer_idx].mlp
            if hasattr(layer, "init"):
                layer.init(self.engine_config.Module_Batching_Config.MoE_decoding_micro_batch_size)

    def _init_ata_comms(self, padding_bsz):
        """Initialize All-to-All communications for EP."""
        try:
            from pplx_kernels.all_to_all import AllToAll
        except ImportError:
            logging.warning("pplx_kernels not available, skipping ATA init")
            return

        in_type = torch.float8_e4m3fn
        out_type = torch.bfloat16
        dp_size = 1
        hidden_size = self.HIDDEN_SIZE
        block_size = 128
        device = self.engine_config.Basic_Config.device_torch

        experts_per_rank = self.NUM_TOTAL_EXPERTS // self.world_size
        num_experts_per_tok = 8
        num_dp = self.world_size // dp_size

        env_max_bsz = os.getenv("BATCHGEN_MAX_RANK_BSZ")
        max_rank_bsz = int(env_max_bsz) if env_max_bsz else padding_bsz

        self.expert_num_tokens = torch.empty(experts_per_rank, dtype=torch.int32, device=device)
        self.expert_x = torch.empty(
            (experts_per_rank, max_rank_bsz * num_dp, hidden_size), dtype=in_type, device=device
        )
        self.expert_x_scale = torch.empty(
            (experts_per_rank, self.expert_x.size(1), (hidden_size + block_size - 1) // block_size),
            dtype=torch.float32, device=device
        )
        self.expert_y = torch.empty_like(self.expert_x, dtype=out_type)
        self.indices = torch.empty(
            (max_rank_bsz, num_experts_per_tok), dtype=torch.uint32, device=device
        )
        self.weights = torch.empty(
            (max_rank_bsz, num_experts_per_tok), dtype=torch.float32, device=device
        )
        self.y = torch.empty((max_rank_bsz, hidden_size), dtype=out_type, device=device)
        self.dp_x = torch.empty((max_rank_bsz, hidden_size), dtype=in_type, device=device)
        self.dp_x_scale = torch.empty(
            (max_rank_bsz, (hidden_size + block_size - 1) // block_size),
            dtype=torch.float32, device=device
        )

        if self.world_size <= 8:
            ata = AllToAll.intranode(
                max_num_tokens=max_rank_bsz, num_experts=self.NUM_TOTAL_EXPERTS,
                experts_per_token=num_experts_per_tok, rank=self.rank,
                world_size=self.world_size, dp_size=dp_size, hidden_dim=hidden_size,
                hidden_dim_bytes=hidden_size * in_type.itemsize,
                hidden_dim_scale_bytes=(hidden_size + block_size - 1) // block_size * 4,
            )
        else:
            ata = AllToAll.internode(
                max_num_tokens=max_rank_bsz, num_experts=self.NUM_TOTAL_EXPERTS,
                experts_per_token=num_experts_per_tok, rank=self.rank,
                world_size=self.world_size, dp_size=dp_size, hidden_dim=hidden_size,
                hidden_dim_bytes=hidden_size * in_type.itemsize,
                hidden_dim_scale_bytes=(hidden_size + block_size - 1) // block_size * 4,
            )

        for layer_idx in range(self.FIRST_K_DENSE, self.model_config.num_hidden_layers):
            layer = self.model.model.layers[layer_idx].mlp
            if hasattr(layer, "init_ata_comm"):
                layer.init_ata_comm(
                    padding_bsz, self.expert_num_tokens, self.expert_x,
                    self.expert_x_scale, self.expert_y, self.indices,
                    self.weights, self.y, self.dp_x, self.dp_x_scale, ata,
                )

    def _load_attn_module(self):
        """Load attention FP8 weights for decode (persistent on GPU)."""
        device = self.engine_config.Basic_Config.device_torch
        for layer_idx in range(len(self.model.model.layers)):
            attn = self.model.model.layers[layer_idx].self_attn
            tensors = self.core_engine.get_tensor(f"attn_{layer_idx}")
            attn.q_a_proj.weight.data = tensors["q_a_proj.weight"].to(device)
            attn.q_b_proj.weight.data = tensors["q_b_proj.weight"].to(device)
            attn.kv_a_proj_with_mqa.weight.data = tensors["kv_a_proj_with_mqa.weight"].to(device)
            attn.kv_b_proj.weight.data = tensors["kv_b_proj.weight"].to(device)
            attn.o_proj.weight.data = tensors["o_proj.weight"].to(device)
            attn.q_a_layernorm.weight.data = tensors["q_a_layernorm.weight"].to(device)
            attn.kv_a_layernorm.weight.data = tensors["kv_a_layernorm.weight"].to(device)

    def _load_shared_expert_module(self):
        """Load shared expert FP8 weights for decode (persistent on GPU)."""
        device = self.engine_config.Basic_Config.device_torch
        for layer_idx in range(self.FIRST_K_DENSE, len(self.model.model.layers)):
            tensors = self.core_engine.get_tensor(f"shared_expert_{layer_idx}")
            shared = self.model.model.layers[layer_idx].mlp.shared_experts
            shared.gate_proj.weight.data = tensors["gate_proj.weight"].to(device)
            shared.up_proj.weight.data = tensors["up_proj.weight"].to(device)
            shared.down_proj.weight.data = tensors["down_proj.weight"].to(device)

    def _load_local_routed_experts(self):
        """Load persistent routed expert FP8 weights for decode.

        Stores weights as flat attributes on placeholder (following GPT-OSS pattern).
        GLM5ExpertWrapper._register_fp8_weights() reads these during _config_expert_module.
        """
        device = self.engine_config.Basic_Config.device_torch
        for routed_expert_idx in self.local_routed_experts:
            tensors = self.core_engine.get_tensor(routed_expert_idx)
            parts = routed_expert_idx.split("_")
            layer_idx = int(parts[2])
            expert_idx = int(parts[3])
            expert = self.model.model.layers[layer_idx].mlp.experts[expert_idx]
            # Store as flat attrs on placeholder (no nn.Module needed)
            expert.fp8_gate = tensors["gate_proj.weight"].to(device)
            expert.fp8_up = tensors["up_proj.weight"].to(device)
            expert.fp8_down = tensors["down_proj.weight"].to(device)
        logging.debug("Local routed experts loaded")

    def _config_attn_module(self):
        """Replace attention modules with GLM5AttnWrapper."""
        start_time = time.perf_counter()
        for layer_idx in range(len(self.model.model.layers)):
            attn_module = self.model.model.layers[layer_idx].self_attn
            if self.engine_config.Basic_Config.gpu_arch == "hopper":
                from batchgen.attention.mla.fa3_backend import (
                    mla_prefill_flashattention3,
                    mla_prefill_flashattention3_w8a16_deepgemm,
                    mla_prefill_flashattention3_prepacked,
                    mla_prefill_flashattention3_w8a16_deepgemm_prepacked,
                )
                from batchgen.attention.mla.flashmla_backend import (
                    mla_decoding_flashmla,
                    mla_decoding_flashmla_attn_mode_3_bf16_with_pagekv,
                    mla_decoding_flashmla_attn_mode_3_fp8_kv_bf16_attn,
                    fused_get_query_states_triton,
                )
                setattr(attn_module, "prefill_attn", types.MethodType(
                    mla_prefill_flashattention3, attn_module))
                setattr(attn_module, "prefill_attn_w8a16", types.MethodType(
                    mla_prefill_flashattention3_w8a16_deepgemm, attn_module))
                setattr(attn_module, "prefill_attn_prepacked", types.MethodType(
                    mla_prefill_flashattention3_prepacked, attn_module))
                setattr(attn_module, "prefill_attn_w8a16_prepacked", types.MethodType(
                    mla_prefill_flashattention3_w8a16_deepgemm_prepacked, attn_module))
                setattr(attn_module, "decoding_attn", types.MethodType(
                    mla_decoding_flashmla, attn_module))
                setattr(attn_module, "decoding_attn_mode_3_bf16", types.MethodType(
                    mla_decoding_flashmla_attn_mode_3_bf16_with_pagekv, attn_module))
                setattr(attn_module, "decoding_attn_mode_3_fp8", types.MethodType(
                    mla_decoding_flashmla_attn_mode_3_fp8_kv_bf16_attn, attn_module))
                setattr(attn_module, "fused_get_query_states_triton", types.MethodType(
                    fused_get_query_states_triton, attn_module))
            elif self.engine_config.Basic_Config.gpu_arch == "ampere":
                from batchgen.attention.mla.fa2_backend import mla_chunked_prefill_flashattention2
                from batchgen.attention.mla.torch_backend import mla_decoding_torch
                setattr(attn_module, "prefill_attn", types.MethodType(
                    mla_chunked_prefill_flashattention2, attn_module))
                setattr(attn_module, "decoding_attn", types.MethodType(
                    mla_decoding_torch, attn_module))
            else:
                raise ValueError(f"Unsupported GPU arch: {self.engine_config.Basic_Config.gpu_arch}")

            # Determine persistence
            persistent = f"attn_{layer_idx}" not in self.weight_copy_task.get("attn", [])

            # Extract FP8 dequant scales from skeleton
            weight_dequant_scales = {}
            prefix = f"model.layers.{layer_idx}.self_attn."
            postfix = ".weight_scale_inv"
            for name, param in self.skeleton_state_dict.items():
                if name.startswith(prefix) and name.endswith(postfix):
                    # Skip indexer scales
                    if ".indexer." in name:
                        continue
                    key = name[len(prefix):]
                    weight_dequant_scales[key] = param.to(
                        self.engine_config.Basic_Config.device_torch
                    )

            wrapper = GLM5AttnWrapper(
                attn_module, layer_idx, self.core_engine,
                self.engine_config, self.model_config,
                persistent, weight_dequant_scales,
            )
            self.model.model.layers[layer_idx].self_attn = wrapper
            if persistent:
                wrapper._register_fp8_weights()

        elapsed = time.perf_counter() - start_time
        logging.debug(f"Attn module config time: {elapsed:.2f}s")

    def _config_expert_module(self):
        """Replace expert modules with GLM5ExpertWrapper."""
        start_time = time.perf_counter()
        mlp_names = ["gate_proj", "up_proj", "down_proj"]
        postfix = ".weight_scale_inv"

        for layer_idx in range(self.FIRST_K_DENSE, len(self.model.model.layers)):
            layer = self.model.model.layers[layer_idx]

            # Shared expert
            shared_persistent = f"shared_expert_{layer_idx}" not in self.weight_copy_task.get("shared_expert", [])
            prefix = f"model.layers.{layer_idx}.mlp.shared_experts."
            shared_scales = {}
            for name in mlp_names:
                key = prefix + name + postfix
                if key in self.skeleton_state_dict:
                    shared_scales[name + postfix] = self.skeleton_state_dict[key].to(
                        self.engine_config.Basic_Config.device_torch
                    )
            layer.mlp.shared_experts = GLM5ExpertWrapper(
                layer.mlp.shared_experts, layer_idx, -1,
                self.core_engine, self.engine_config, self.model_config,
                shared_persistent, shared_scales, is_fp8=self.is_fp8_experts,
            )
            if shared_persistent:
                layer.mlp.shared_experts._register_fp8_weights()

            # Routed experts — wrap placeholders directly (no nn.Module needed)
            local_set = set(self.local_routed_experts) if hasattr(self, 'local_routed_experts') else set()
            for expert_idx in range(len(layer.mlp.experts)):
                routed_key = f"routed_expert_{layer_idx}_{expert_idx}"
                persistent = routed_key not in self.weight_copy_task.get("routed_expert", [])

                prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}."
                expert_scales = {}
                for name in mlp_names:
                    key = prefix + name + postfix
                    if key in self.skeleton_state_dict:
                        expert_scales[name + postfix] = self.skeleton_state_dict[key].to(
                            self.engine_config.Basic_Config.device_torch
                        )
                layer.mlp.experts[expert_idx] = GLM5ExpertWrapper(
                    layer.mlp.experts[expert_idx], layer_idx, expert_idx,
                    self.core_engine, self.engine_config, self.model_config,
                    persistent, expert_scales, is_fp8=self.is_fp8_experts,
                )
                # Only register weights for experts that had weights loaded
                if persistent and routed_key in local_set:
                    layer.mlp.experts[expert_idx]._register_fp8_weights()

        elapsed = time.perf_counter() - start_time
        logging.debug(f"Expert module config time: {elapsed:.2f}s")

    def _setup_decode_moe(self, comm):
        """Replace Glm5MoE with Glm5MoEDecode for EP decode path.

        Creates Glm5MoEDecode per MoE layer, transfers gate, shared_experts,
        and all expert wrappers from the old Glm5MoE. Pointer arrays for
        grouped FP8 GEMM are built later in _init_mode_decoding() via init().
        """
        device = self.engine_config.Basic_Config.device_torch
        NUM_EXPERT_PER_RANK = self.NUM_TOTAL_EXPERTS // self.world_size

        for layer_idx in range(self.FIRST_K_DENSE, self.model_config.num_hidden_layers):
            old_moe = self.model.model.layers[layer_idx].mlp

            decode_moe = Glm5MoEDecode(self.loaded_model_config, comm=comm)

            # Transfer gate and shared experts
            decode_moe.gate = old_moe.gate
            decode_moe.shared_experts = old_moe.shared_experts

            # Set EP range
            decode_moe.routed_expert_start_idx = self.global_rank * NUM_EXPERT_PER_RANK
            decode_moe.routed_expert_end_idx = (self.global_rank + 1) * NUM_EXPERT_PER_RANK
            decode_moe.experts_per_rank = NUM_EXPERT_PER_RANK
            decode_moe.num_persistent_local_experts = self.num_local_expert_per_layer

            # Transfer all expert wrappers (persistent + non-persistent)
            for expert_idx in range(self.NUM_TOTAL_EXPERTS):
                decode_moe.experts[expert_idx] = old_moe.experts[expert_idx]

            # Propagate offloading flag
            decode_moe.enable_ep_offloading = self.enable_ep_offloading

            decode_moe.to(device)
            self.model.model.layers[layer_idx].mlp = decode_moe

        if self.rank == 0:
            logging.info(
                f"[DECODE] Set up Glm5MoEDecode for {self.model_config.num_hidden_layers - self.FIRST_K_DENSE} "
                f"MoE layers (experts_per_rank={NUM_EXPERT_PER_RANK}, "
                f"offloading={self.enable_ep_offloading})"
            )

    def _load_model_skeleton(self):
        """Load skeleton weights as-is (no CPU dequant). FP8 dequant happens on-the-fly."""
        for key, param in self.model.named_parameters():
            if key in self.state_dict_name_map:
                continue
            if key in self.skeleton_state_dict:
                param.data = self.skeleton_state_dict[key]

        skeleton_size = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        ) / (1024**3)
        if self.rank == 0:
            logging.info(f"Model skeleton size: {skeleton_size:.2f} GB")

    def _setup_fp8_scales(self):
        """Attach FP8 scale tensors to indexer and dense MLP for on-the-fly dequant."""
        device = self.engine_config.Basic_Config.device_torch
        for layer_idx in range(self.model_config.num_hidden_layers):
            indexer = self.model.model.layers[layer_idx].self_attn.indexer
            for proj, attr in [("wk", "wk_scale"), ("wq_b", "wq_b_scale")]:
                key = f"model.layers.{layer_idx}.self_attn.indexer.{proj}.weight_scale_inv"
                if key in self.dequant_scale:
                    setattr(indexer, attr, self.dequant_scale[key].to(device))
        for layer_idx in range(self.FIRST_K_DENSE):
            mlp = self.model.model.layers[layer_idx].mlp
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                key = f"model.layers.{layer_idx}.mlp.{proj}.weight_scale_inv"
                if key in self.dequant_scale:
                    setattr(mlp, f"{proj.split('_')[0]}_scale",
                            self.dequant_scale[key].to(device))

    def _lm_head_forward_pre_hook(self, module, input):
        return input[0][:, -1, :].unsqueeze(1)

    def _config_lm_head_hook(self):
        self.model.lm_head.register_forward_pre_hook(self._lm_head_forward_pre_hook)

    def _extract_dequantize_scale(self):
        self.dequant_scale = {}
        for key, param in self.skeleton_state_dict.items():
            if "weight_scale_inv" in key:
                self.dequant_scale[key] = param
