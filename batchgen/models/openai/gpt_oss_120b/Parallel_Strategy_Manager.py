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

"""Parallel Strategy Manager for GPT-OSS-120B.

Handles model skeleton creation, weight loading, and execution strategy.
Optimized for single H20 GPU with MXFP4 quantization.
"""

import gc
import logging
import time
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist

from .modeling_gpt_oss import GptOssForCausalLM
from batchgen.models.wrappers import GptOssExpertWrapper, GptOssAttnWrapper


def _build_gpt_oss_name_mapping() -> Dict[str, str]:
    """Build mapping from original GPT-OSS checkpoint names to HuggingFace names.

    Original GPT-OSS naming (0-indexed):
    - embedding.weight → model.embed_tokens.weight
    - unembedding.weight → lm_head.weight
    - block.{N}.attn.norm.scale → model.layers.{N}.input_layernorm.weight
    - block.{N}.mlp.norm.scale → model.layers.{N}.post_attention_layernorm.weight
    - block.{N}.attn.out.weight/bias → model.layers.{N}.self_attn.o_proj.weight/bias
    - block.{N}.mlp.gate.weight/bias → model.layers.{N}.mlp.router.weight/bias
    - final_norm.scale → model.norm.weight

    Note: QKV weights (block.{N}.attn.qkv.*) need special handling - see _load_model_skeleton()

    Returns:
        Dict mapping original checkpoint names to HuggingFace names
    """
    mapping = {
        "embedding.weight": "model.embed_tokens.weight",
        "unembedding.weight": "lm_head.weight",
        "final_norm.scale": "model.norm.weight",
    }

    # Build per-layer mappings (36 layers, 0-indexed in both)
    for layer_idx in range(36):
        # Layer norms
        mapping[f"block.{layer_idx}.attn.norm.scale"] = f"model.layers.{layer_idx}.input_layernorm.weight"
        mapping[f"block.{layer_idx}.mlp.norm.scale"] = f"model.layers.{layer_idx}.post_attention_layernorm.weight"

        # Attention output projection
        mapping[f"block.{layer_idx}.attn.out.weight"] = f"model.layers.{layer_idx}.self_attn.o_proj.weight"
        mapping[f"block.{layer_idx}.attn.out.bias"] = f"model.layers.{layer_idx}.self_attn.o_proj.bias"

        # Router (gate)
        mapping[f"block.{layer_idx}.mlp.gate.weight"] = f"model.layers.{layer_idx}.mlp.router.weight"
        mapping[f"block.{layer_idx}.mlp.gate.bias"] = f"model.layers.{layer_idx}.mlp.router.bias"

    return mapping


def _build_reverse_mapping(mapping: Dict[str, str]) -> Dict[str, str]:
    """Build reverse mapping from HuggingFace names to original names."""
    return {v: k for k, v in mapping.items()}


class GptOssParallelStrategyManager:
    """Manage parallel execution strategy for GPT-OSS-120B.

    For single H20 GPU deployment:
    - All 128 experts are local
    - No tensor parallelism
    - MXFP4 dequantization for expert weights
    - BF16 attention weights (not quantized)
    """

    # Class-level name mapping (built once)
    _name_mapping = None

    @classmethod
    def get_name_mapping(cls) -> Dict[str, str]:
        """Get the original→HuggingFace name mapping (cached)."""
        if cls._name_mapping is None:
            cls._name_mapping = _build_gpt_oss_name_mapping()
        return cls._name_mapping

    def __init__(
        self,
        hf_model_config,
        engine_config,
        model_config,
        core_engine,
        skeleton_state_dict,
        local_rank: int,
        global_rank: int,
        world_size: int,
    ):
        self.hf_model_config = hf_model_config
        self.engine_config = engine_config
        self.model_config = model_config
        self.core_engine = core_engine
        self.skeleton_state_dict = skeleton_state_dict
        self.weight_copy_task = {}

        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.rank = global_rank

    def configure_prefill(self) -> Tuple:
        """Configure model skeleton for prefill phase.

        Returns:
            Tuple of (model, weight_copy_task)
        """
        start_time = time.perf_counter()
        timings = {}

        # Step 1: Set phase
        self.hf_model_config.phase = "prefill"

        # Step 2: Initialize model
        step_start = time.perf_counter()
        self.model = GptOssForCausalLM(self.hf_model_config)
        timings["model_init"] = time.perf_counter() - step_start

        # Step 3: Initialize data structures
        self.state_dict_name_map = {}
        self.weight_copy_task = {}
        self.weight_copy_task["attn"] = []
        self.weight_copy_task["routed_expert"] = []

        # Step 4: Build weight copy task mappings
        step_start = time.perf_counter()
        for layer_idx in range(self.model_config.num_hidden_layers):
            # Attention parameters (BF16, not quantized)
            for name, _ in self.model.model.layers[layer_idx].self_attn.named_parameters():
                tensor_full_name = f"model.layers.{layer_idx}.self_attn.{name}"
                self.state_dict_name_map[tensor_full_name] = {
                    "module_key": f"attn_{layer_idx}",
                    "tensor_key": name,
                }
            self.weight_copy_task["attn"].append(f"attn_{layer_idx}")

            # Routed experts (MXFP4 quantized)
            # GPT-OSS uses SHARED weights per layer - all 128 experts share the same
            # mlp1_weight and mlp2_weight tensors. Each expert wrapper slices its portion.
            # Weight copy task uses per-layer key (not per-expert)
            self.weight_copy_task["routed_expert"].append(f"routed_expert_{layer_idx}")

        timings["mapping_build"] = time.perf_counter() - step_start

        # Step 5: Load model skeleton and configure modules
        step_start = time.perf_counter()
        self._load_model_skeleton()
        timings["skeleton_load"] = time.perf_counter() - step_start

        step_start = time.perf_counter()
        self._config_attn_module()
        timings["attn_config"] = time.perf_counter() - step_start

        step_start = time.perf_counter()
        self._config_expert_module()
        timings["expert_config"] = time.perf_counter() - step_start

        step_start = time.perf_counter()
        self._config_lm_head_hook()
        timings["lm_head_config"] = time.perf_counter() - step_start

        self.model.eval()
        self.model.to(self.engine_config.Basic_Config.device_torch)

        total_time = time.perf_counter() - start_time
        logging.info(f"Model configuration complete in {total_time:.2f}s")
        logging.info(f"Timings: {timings}")

        return self.model, self.weight_copy_task

    def _load_model_skeleton(self):
        """Load non-expert weights (embeddings, attention, norms, lm_head).

        Handles name mapping from original GPT-OSS checkpoint format to HuggingFace format.
        Also handles QKV weight splitting since original uses combined qkv weights.
        """
        logging.info("Loading model skeleton...")

        # Debug: Log available keys in skeleton_state_dict
        skeleton_keys = list(self.skeleton_state_dict.keys())
        logging.info(f"Skeleton state dict has {len(skeleton_keys)} keys")
        if skeleton_keys:
            logging.info(f"First 20 skeleton keys: {skeleton_keys[:20]}")
            # Check for specific keys we expect
            for check_key in ["embedding.weight", "unembedding.weight", "final_norm.scale",
                              "block.0.attn.qkv.weight", "block.0.attn.norm.scale"]:
                if check_key in self.skeleton_state_dict:
                    logging.info(f"Found key: {check_key}")
                else:
                    logging.warning(f"Missing expected key: {check_key}")

        # Build reverse mapping: HuggingFace name → original checkpoint name
        fwd_mapping = self.get_name_mapping()
        rev_mapping = _build_reverse_mapping(fwd_mapping)

        # Model config for QKV splitting
        num_q_heads = self.hf_model_config.num_attention_heads  # 64
        num_kv_heads = self.hf_model_config.num_key_value_heads  # 8
        head_dim = self.hf_model_config.head_dim  # 64

        loaded_count = 0
        missing_count = 0
        missing_keys = []

        for name, param in self.model.named_parameters():
            if "experts" in name:
                continue  # Skip expert weights (loaded separately with MXFP4)

            # Try direct match first
            if name in self.skeleton_state_dict:
                param.data.copy_(self.skeleton_state_dict[name])
                loaded_count += 1
                continue

            # Try reverse mapping (HuggingFace → original)
            original_name = rev_mapping.get(name)
            if original_name and original_name in self.skeleton_state_dict:
                param.data.copy_(self.skeleton_state_dict[original_name])
                loaded_count += 1
                continue

            # Handle QKV weight splitting
            # Original: block.{N}.attn.qkv.weight → q_proj, k_proj, v_proj
            if "self_attn.q_proj" in name or "self_attn.k_proj" in name or "self_attn.v_proj" in name:
                # Extract layer index
                parts = name.split(".")
                layer_idx = int(parts[2])
                suffix = "weight" if "weight" in name else "bias"
                qkv_key = f"block.{layer_idx}.attn.qkv.{suffix}"

                if qkv_key in self.skeleton_state_dict:
                    qkv_tensor = self.skeleton_state_dict[qkv_key]

                    # QKV layout: [q_heads * head_dim + 2 * kv_heads * head_dim, hidden]
                    # Split into Q, K, V
                    q_size = num_q_heads * head_dim
                    kv_size = num_kv_heads * head_dim

                    if "q_proj" in name:
                        param.data.copy_(qkv_tensor[:q_size])
                    elif "k_proj" in name:
                        param.data.copy_(qkv_tensor[q_size:q_size + kv_size])
                    elif "v_proj" in name:
                        param.data.copy_(qkv_tensor[q_size + kv_size:q_size + 2 * kv_size])

                    loaded_count += 1
                    continue

            # Not found
            missing_count += 1
            if missing_count <= 10:
                missing_keys.append(name)

        if missing_keys:
            logging.warning(f"Missing skeleton weights (first 10): {missing_keys}")
        if missing_count > 10:
            logging.warning(f"... and {missing_count - 10} more missing weights")

        logging.info(f"Model skeleton loaded: {loaded_count} loaded, {missing_count} missing")

    def _config_attn_module(self):
        """Configure attention modules with BatchGen wrappers."""
        logging.info("Configuring attention modules...")

        for layer_idx in range(self.model_config.num_hidden_layers):
            attn = self.model.model.layers[layer_idx].self_attn

            # Wrap attention for BatchGen execution
            # GptOssAttnWrapper handles GQA and alternating sliding/full attention
            wrapped_attn = GptOssAttnWrapper(
                module=attn,
                layer_idx=layer_idx,
                core_engine=self.core_engine,
                engine_config=self.engine_config,
                model_config=self.model_config,
            )
            self.model.model.layers[layer_idx].self_attn = wrapped_attn

        logging.info(f"Configured {self.model_config.num_hidden_layers} attention modules")

    def _config_expert_module(self):
        """Configure expert modules with MXFP4 dequantization."""
        logging.info("Configuring expert modules...")

        for layer_idx in range(self.model_config.num_hidden_layers):
            for expert_idx in range(self.model_config.num_local_experts):
                expert = self.model.model.layers[layer_idx].mlp.experts[expert_idx]

                # Wrap expert for BatchGen execution with MXFP4 support
                # GptOssExpertWrapper handles MXFP4 dequantization internally
                wrapped_expert = GptOssExpertWrapper(
                    module=expert,
                    layer_idx=layer_idx,
                    expert_idx=expert_idx,
                    core_engine=self.core_engine,
                    engine_config=self.engine_config,
                    model_config=self.model_config,
                )
                self.model.model.layers[layer_idx].mlp.experts[expert_idx] = wrapped_expert

        total_experts = self.model_config.num_hidden_layers * self.model_config.num_local_experts
        logging.info(f"Configured {total_experts} expert modules")

    def _config_lm_head_hook(self):
        """Configure LM head for output processing."""
        logging.info("Configuring LM head...")
        # LM head is in BF16, no special handling needed for MXFP4

    def configure_decoding(self) -> Tuple:
        """Configure model for decoding phase.

        Same as prefill for GPT-OSS on single GPU.
        """
        return self.configure_prefill()

    def get_weight_copy_task(self) -> Dict[str, List[str]]:
        """Return the weight copy task mapping."""
        return self.weight_copy_task

    def get_state_dict_name_map(self) -> Dict[str, Dict[str, str]]:
        """Return the state dict name mapping."""
        return self.state_dict_name_map
