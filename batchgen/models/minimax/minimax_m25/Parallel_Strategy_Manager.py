# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
# ---------------------------------------------------------------------------- #

"""MiniMax-M2.5 Parallel Strategy Manager for BatchGen.

Handles model initialization, weight loading, and EP configuration for MiniMax-M2.5.

Key differences from Kimi K2.5 PSM:
- GQA attention (not MLA) — loads q_proj, k_proj, v_proj, o_proj, q_norm, k_norm
- FP8 expert weights (not INT4) — uses deepseek_v3_dequantization
- 256 routed experts (not 384), no shared experts
- All 62 layers are MoE (no first_k_dense_replace)
- BF16 attention (same as K2.5)
"""

from .model import MiniMaxM25Model
from .wrappers import MiniMaxM25ExpertWrapper, MiniMaxM25AttnWrapper
import logging
import types
import time
import torch
import gc
import os

NUM_TOTAL_EXPERTS = 256


class MiniMaxM25ParallelStrategyManager:
    def __init__(self, loaded_model_config, engine_config, model_config,
                 core_engine, skeleton_state_dict, local_rank, global_rank, world_size):
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
        self.model = None
        self.enable_ep_offloading = False

    def configure_prefill(self):
        """Configure model for prefill (pure DP)."""
        start_time = time.perf_counter()
        self.loaded_model_config.phase = "prefill"

        self.model = MiniMaxM25Model(self.loaded_model_config)

        self.weight_copy_task = {"attn": [], "routed_expert": []}

        for layer_idx in range(self.model_config.num_hidden_layers):
            self.weight_copy_task["attn"].append(f"attn_{layer_idx}")
            for expert_idx in range(NUM_TOTAL_EXPERTS):
                self.weight_copy_task["routed_expert"].append(
                    f"routed_expert_{layer_idx}_{expert_idx}"
                )

        self._load_model_skeleton()
        self._config_attn_module()
        self._config_expert_module()
        self._config_unembedding_hook()

        self.model.eval()
        self.model.to(self.engine_config.Basic_Config.device_torch)

        total_time = time.perf_counter() - start_time
        if self.rank == 0:
            logging.info(f"[PREFILL] Model configured in {total_time:.2f}s")

        return self.model, self.weight_copy_task

    def configure_decoding(self, padding_bsz=None, comm=None):
        """Configure model for decoding: DP + EP."""
        self.loaded_model_config.phase = "decode"
        self.loaded_model_config._attn_implementation = "eager"
        self.loaded_model_config.ep_size = self.world_size

        device = self.engine_config.Basic_Config.device_torch

        if self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        torch.cuda.empty_cache()

        with torch.device('cpu'):
            self.model = MiniMaxM25Model(self.loaded_model_config, comm)

        # Inject comm into MoE layers
        for layer_idx in range(self.model_config.num_hidden_layers):
            moe = self.model.layers[layer_idx].mlp
            moe.comm = comm
            moe.device = device

        self.weight_copy_task = {"attn": [], "routed_expert": []}
        self.local_routed_experts = []
        self.host_routed_experts = []

        NUM_EXPERT_PER_RANK = NUM_TOTAL_EXPERTS // self.world_size

        if self.world_size > 8:
            self.enable_ep_offloading = False
            NUM_LOCAL_EXPERT_PER_LAYER = NUM_EXPERT_PER_RANK
        elif self.engine_config.EP_Config.enable_offloading:
            offload_ratio = self.engine_config.EP_Config.offloading_ratio
            self.enable_ep_offloading = True
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

        for layer_idx in range(self.model_config.num_hidden_layers):
            for expert_idx in range(routed_expert_gpu_start_idx, routed_expert_gpu_end_idx):
                self.local_routed_experts.append(f"routed_expert_{layer_idx}_{expert_idx}")
            for expert_idx in range(routed_expert_gpu_end_idx, routed_expert_host_end_idx):
                self.host_routed_experts.append(f"routed_expert_{layer_idx}_{expert_idx}")

        self.weight_copy_task["routed_expert"] = self.host_routed_experts

        self._load_model_skeleton()
        self._load_local_routed_experts()
        self._load_attn_module()
        self._config_attn_module()
        self._config_expert_module()
        self._config_unembedding_hook()

        # Set persistent/non-persistent expert lists and offloading flag per layer
        for layer_idx in range(self.model_config.num_hidden_layers):
            layer = self.model.layers[layer_idx]
            layer.mlp.persistent_expert_ids = list(
                range(routed_expert_gpu_start_idx, routed_expert_gpu_end_idx)
            )
            layer.mlp.nonpersistent_expert_ids = list(
                range(routed_expert_gpu_end_idx, routed_expert_host_end_idx)
            )
            layer.mlp.enable_ep_offloading = self.enable_ep_offloading

        self.model.eval()
        self.model.to(device)

        self._init_mode_decoding()
        effective_padding_bsz = padding_bsz if padding_bsz is not None else 128
        self._init_decoding_padding_bsz(effective_padding_bsz)

        return self.model, self.weight_copy_task

    def _init_decoding_padding_bsz(self, padding_bsz):
        env_max_bsz = os.getenv("BATCHGEN_MAX_RANK_BSZ")
        max_rank_bsz = int(env_max_bsz) if env_max_bsz else padding_bsz
        for layer_idx in range(self.model_config.num_hidden_layers):
            self.model.layers[layer_idx].mlp.init_num_tokens(max_rank_bsz)

    def set_num_tokens_per_rank(self, num_tokens_per_rank):
        for layer_idx in range(self.model_config.num_hidden_layers):
            self.model.layers[layer_idx].mlp.set_num_tokens_per_rank(num_tokens_per_rank)

    def _init_mode_decoding(self):
        if self.enable_ep_offloading:
            return
        for layer_idx in range(self.model_config.num_hidden_layers):
            self.model.layers[layer_idx].mlp.init(
                self.engine_config.Module_Batching_Config.MoE_decoding_micro_batch_size
            )

    def _load_attn_module(self):
        """Load BF16 GQA attention weights from core engine."""
        for layer_idx in range(len(self.model.layers)):
            attn = self.model.layers[layer_idx].self_attn
            tensors = self.core_engine.get_tensor(f"attn_{layer_idx}")
            device = self.engine_config.Basic_Config.device_torch
            attn.q_proj.weight.data = tensors["q_proj.weight"].to(device)
            attn.k_proj.weight.data = tensors["k_proj.weight"].to(device)
            attn.v_proj.weight.data = tensors["v_proj.weight"].to(device)
            attn.o_proj.weight.data = tensors["o_proj.weight"].to(device)
            attn.q_norm.weight.data = tensors["q_norm.weight"].to(device)
            attn.k_norm.weight.data = tensors["k_norm.weight"].to(device)

    def _load_local_routed_experts(self):
        """Load FP8 weights for persistent routed experts."""
        for routed_expert_idx in self.local_routed_experts:
            tensors = self.core_engine.get_tensor(routed_expert_idx)
            layer_idx = int(routed_expert_idx.split("_")[2])
            global_expert_idx = int(routed_expert_idx.split("_")[3])
            expert = self.model.layers[layer_idx].mlp.experts[global_expert_idx]
            if expert is None:
                continue
            for name, param in expert.named_parameters():
                if name in tensors:
                    param.data = tensors[name]

    def _load_model_skeleton(self):
        """Load skeleton weights (norms, embeddings, router, correction bias)."""
        for key, param in self.model.named_parameters():
            if key in self.skeleton_state_dict:
                param.data = self.skeleton_state_dict[key]

        # Also load buffers (e_score_correction_bias)
        for key, buf in self.model.named_buffers():
            if key in self.skeleton_state_dict:
                buf.data = self.skeleton_state_dict[key]

        if self.rank == 0:
            size_gb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**3)
            logging.info(f"Model skeleton size: {size_gb:.2f} GB")

    def _config_attn_module(self):
        """Configure GQA attention wrappers with FA3/FA2."""
        for layer_idx in range(len(self.model.layers)):
            attn_module = self.model.layers[layer_idx].self_attn

            # Inject GQA prefill/decode methods
            if self.engine_config.Basic_Config.gpu_arch == "hopper":
                from batchgen.attention.gqa.fa_prefill import gqa_prefill_fa3, gqa_prefill_fa3_prepacked
                from batchgen.attention.gqa.fa_decode import gqa_decode_fa3_paged
                setattr(attn_module, "prefill_attn_gqa",
                        types.MethodType(gqa_prefill_fa3, attn_module))
                setattr(attn_module, "prefill_attn_gqa_prepacked",
                        types.MethodType(gqa_prefill_fa3_prepacked, attn_module))
                setattr(attn_module, "decoding_attn_gqa_paged",
                        types.MethodType(gqa_decode_fa3_paged, attn_module))
            elif self.engine_config.Basic_Config.gpu_arch == "ampere":
                from batchgen.attention.gqa.fa_prefill import gqa_prefill_fa2
                from batchgen.attention.gqa.fa_decode import gqa_decode_fa2_paged
                setattr(attn_module, "prefill_attn_gqa",
                        types.MethodType(gqa_prefill_fa2, attn_module))
                setattr(attn_module, "decoding_attn_gqa_paged",
                        types.MethodType(gqa_decode_fa2_paged, attn_module))

            persistent = f"attn_{layer_idx}" not in self.weight_copy_task.get("attn", [])
            wrapper = MiniMaxM25AttnWrapper(
                attn_module, layer_idx, self.core_engine, self.engine_config,
                self.model_config, persistent,
            )
            self.model.layers[layer_idx].self_attn = wrapper

    def _config_expert_module(self):
        """Replace expert modules with FP8 wrappers."""
        for layer_idx in range(len(self.model.layers)):
            layer = self.model.layers[layer_idx]
            for expert_idx in range(len(layer.mlp.experts)):
                expert = layer.mlp.experts[expert_idx]
                if expert is None:
                    continue
                module_key = f"routed_expert_{layer_idx}_{expert_idx}"
                persistent = module_key not in self.weight_copy_task.get("routed_expert", [])

                weight_dequant_scale = self._extract_dequant_scales(layer_idx, expert_idx)

                wrapper = MiniMaxM25ExpertWrapper(
                    expert, layer_idx, expert_idx, self.core_engine,
                    self.engine_config, self.model_config, persistent,
                    weight_dequant_scale=weight_dequant_scale,
                )
                if persistent:
                    wrapper._register_fp8_weights()
                    for key, value in wrapper.weight_dequant_scale.items():
                        wrapper.weight_dequant_scale[key] = value.to(
                            self.engine_config.Basic_Config.device_torch
                        )
                layer.mlp.experts[expert_idx] = wrapper

    def _extract_dequant_scales(self, layer_idx, expert_idx):
        """Extract FP8 dequant scale factors for an expert.

        HF checkpoint naming: w1=gate_proj, w2=down_proj, w3=up_proj.
        Wrapper expects: gate_proj.weight_scale_inv, up_proj.weight_scale_inv, down_proj.weight_scale_inv.
        """
        scales = {}
        prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}"
        # w1 -> gate_proj, w2 -> down_proj, w3 -> up_proj
        hf_to_bg = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
        for hf_name, bg_name in hf_to_bg.items():
            full_key = f"{prefix}.{hf_name}.weight_scale_inv"
            if full_key in self.skeleton_state_dict:
                scales[f"{bg_name}.weight_scale_inv"] = self.skeleton_state_dict[full_key]
        return scales

    def _unembedding_forward_pre_hook(self, module, input):
        return input[0][:, -1, :].unsqueeze(1)

    def _config_unembedding_hook(self):
        self.model.unembedding.register_forward_pre_hook(self._unembedding_forward_pre_hook)
