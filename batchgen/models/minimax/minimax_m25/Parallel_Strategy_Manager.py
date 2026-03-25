# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
# ---------------------------------------------------------------------------- #

"""MiniMax-M2.5 Parallel Strategy Manager for BatchGen.

Handles model initialization, weight loading, and EP configuration for MiniMax-M2.5.

Core engine tensor keys (from parameter_server.py):
  attn_{L}:              q_proj.weight, k_proj.weight, v_proj.weight, o_proj.weight,
                         q_norm.weight, k_norm.weight
  routed_expert_{L}_{E}: w1.weight, w1.weight_scale_inv  (gate_proj)
                         w2.weight, w2.weight_scale_inv  (down_proj)
                         w3.weight, w3.weight_scale_inv  (up_proj)

Skeleton state dict uses HF checkpoint keys (block_sparse_moe), while model
uses `mlp` as the nn.Module attribute name. _model_key_to_ckpt_key() maps between them.
"""

from .model import MiniMaxM25
from .wrappers import MiniMaxM25ExpertWrapper, MiniMaxM25AttnWrapper
import logging
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
        """Configure model for prefill (pure DP).

        All attn and expert modules are non-persistent (loaded from core_engine
        buffers on demand). Weight copy tasks include all modules.
        """
        start_time = time.perf_counter()
        self.loaded_model_config.phase = "prefill"

        self.model = MiniMaxM25(self.loaded_model_config)

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
            self.model = MiniMaxM25(self.loaded_model_config)

        # Inject comm into MoE layers
        for layer_idx in range(self.model_config.num_hidden_layers):
            moe = self.model.model.layers[layer_idx].mlp
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

        # 1. Load skeleton (norms, embeddings, gate, correction_bias)
        self._load_model_skeleton()
        # 2. Wrap attention modules
        self._config_attn_module()
        # 3. Load FP8 attention weights into wrappers (persistent for decode)
        self._load_attn_module()
        # 4. Wrap expert modules (placeholders → wrappers)
        self._config_expert_module()
        # 5. Load FP8 weights into persistent expert wrappers
        self._load_local_routed_experts()
        # 6. Configure unembedding hook
        self._config_unembedding_hook()

        # Set persistent/non-persistent expert lists and offloading flag per layer
        for layer_idx in range(self.model_config.num_hidden_layers):
            layer = self.model.model.layers[layer_idx]
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
            self.model.model.layers[layer_idx].mlp.init_num_tokens(max_rank_bsz)

    def set_num_tokens_per_rank(self, num_tokens_per_rank):
        for layer_idx in range(self.model_config.num_hidden_layers):
            self.model.model.layers[layer_idx].mlp.set_num_tokens_per_rank(num_tokens_per_rank)

    def _init_mode_decoding(self):
        if self.enable_ep_offloading:
            return
        for layer_idx in range(self.model_config.num_hidden_layers):
            self.model.model.layers[layer_idx].mlp.init(
                self.engine_config.Module_Batching_Config.MoE_decoding_micro_batch_size
            )

    def _load_attn_module(self):
        """Load FP8 attention weights + scales from core engine into attn wrappers.

        Must be called AFTER _config_attn_module() so self_attn is a wrapper.
        """
        for layer_idx in range(len(self.model.model.layers)):
            wrapper = self.model.model.layers[layer_idx].self_attn
            tensors = self.core_engine.get_tensor(f"attn_{layer_idx}")
            wrapper._register_fp8_weights(tensors)

    def _load_local_routed_experts(self):
        """Load FP8 weights + scales for persistent routed expert wrappers.

        Must be called AFTER _config_expert_module() so experts[idx] is a wrapper.
        Core engine tensors: w1.weight, w1.weight_scale_inv, w2.weight, etc.
        """
        device = self.engine_config.Basic_Config.device_torch
        for routed_expert_key in self.local_routed_experts:
            tensors = self.core_engine.get_tensor(routed_expert_key)
            layer_idx = int(routed_expert_key.split("_")[2])
            global_expert_idx = int(routed_expert_key.split("_")[3])
            wrapper = self.model.model.layers[layer_idx].mlp.experts[global_expert_idx]
            # Move tensors to device and register on wrapper
            device_tensors = {k: v.to(device) for k, v in tensors.items()}
            wrapper._register_fp8_weights(device_tensors)

    def _model_key_to_ckpt_key(self, key):
        """Map model parameter name to HF checkpoint key.

        Model uses `mlp` (nn.Module attribute), checkpoint uses `block_sparse_moe`.
        """
        return key.replace(".mlp.", ".block_sparse_moe.")

    def _load_model_skeleton(self):
        """Load skeleton weights (norms, embeddings, router, correction bias).

        Skeleton state dict uses HF checkpoint keys (block_sparse_moe).
        Model named_parameters() uses `mlp`. Map between them.
        """
        loaded = 0
        not_found = []
        for key, param in self.model.named_parameters():
            ckpt_key = self._model_key_to_ckpt_key(key)
            found_key = None
            if ckpt_key in self.skeleton_state_dict:
                found_key = ckpt_key
            elif key in self.skeleton_state_dict:
                found_key = key
            if found_key is not None:
                raw_tensor = self.skeleton_state_dict[found_key]
                # Log gate weight dtype to verify checkpoint format
                if "gate.weight" in key and self.rank == 0:
                    logging.warning(
                        f"[SKELETON] {key}: raw_dtype={raw_tensor.dtype}, "
                        f"shape={list(raw_tensor.shape)}, "
                        f"param_dtype={param.dtype}")
                param.data = raw_tensor
                loaded += 1
            else:
                not_found.append(key)

        # Also load buffers (e_score_correction_bias)
        for key, buf in self.model.named_buffers():
            ckpt_key = self._model_key_to_ckpt_key(key)
            if ckpt_key in self.skeleton_state_dict:
                buf.data = self.skeleton_state_dict[ckpt_key]
                loaded += 1
            elif key in self.skeleton_state_dict:
                buf.data = self.skeleton_state_dict[key]
                loaded += 1
            else:
                not_found.append(key)

        if self.rank == 0:
            size_gb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**3)
            logging.info(f"Model skeleton: {loaded} tensors loaded, {size_gb:.2f} GB")

            logging.info(f"Skeleton state_dict has {len(self.skeleton_state_dict)} keys")

            if not_found:
                logging.warning(f"Skeleton: {len(not_found)} model params not found in skeleton")
                for k in not_found[:3]:
                    logging.warning(f"  NOT FOUND: {k}")
                if len(not_found) > 3:
                    logging.warning(f"  ... and {len(not_found) - 3} more")

    def _config_attn_module(self):
        """Configure GQA attention wrappers."""
        for layer_idx in range(len(self.model.model.layers)):
            attn_module = self.model.model.layers[layer_idx].self_attn

            persistent = f"attn_{layer_idx}" not in self.weight_copy_task.get("attn", [])
            wrapper = MiniMaxM25AttnWrapper(
                attn_module, layer_idx, self.core_engine, self.engine_config,
                self.model_config, persistent,
            )
            self.model.model.layers[layer_idx].self_attn = wrapper

    def _config_expert_module(self):
        """Replace expert placeholders with FP8 wrappers.

        Wrappers are created around placeholders. For persistent experts,
        FP8 weights are loaded later by _load_local_routed_experts().
        For non-persistent experts, weights are loaded from core_engine on demand.
        """
        for layer_idx in range(len(self.model.model.layers)):
            layer = self.model.model.layers[layer_idx]
            for expert_idx in range(len(layer.mlp.experts)):
                expert = layer.mlp.experts[expert_idx]
                if expert is None:
                    continue
                module_key = f"routed_expert_{layer_idx}_{expert_idx}"
                persistent = module_key not in self.weight_copy_task.get("routed_expert", [])

                wrapper = MiniMaxM25ExpertWrapper(
                    expert, layer_idx, expert_idx, self.core_engine,
                    self.engine_config, self.model_config, persistent,
                )
                layer.mlp.experts[expert_idx] = wrapper

    def _unembedding_forward_pre_hook(self, module, input):
        return input[0][:, -1, :].unsqueeze(1)

    def _config_unembedding_hook(self):
        self.model.lm_head.register_forward_pre_hook(self._unembedding_forward_pre_hook)
