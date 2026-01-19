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

from .model import GptOss
from .wrappers import GptOssExpertWrapper, GptOssAttnWrapper


class GptOssParallelStrategyManager:
    """Manage parallel execution strategy for GPT-OSS-120B.

    For single H20 GPU deployment:
    - All 128 experts are local
    - No tensor parallelism
    - MXFP4 dequantization for expert weights
    - BF16 attention weights (not quantized)
    """

    def __init__(
        self,
        loaded_model_config,
        engine_config,
        model_config,
        core_engine,
        skeleton_state_dict,
        local_rank: int,
        global_rank: int,
        world_size: int,
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

    def configure_prefill(self) -> Tuple:
        """Configure model skeleton for prefill phase.

        Returns:
            Tuple of (model, weight_copy_task)
        """
        start_time = time.perf_counter()
        timings = {}

        # Step 1: Set phase
        self.loaded_model_config.phase = "prefill"

        # Step 2: Initialize model
        step_start = time.perf_counter()
        self.model = GptOss(self.loaded_model_config)
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
            for expert_idx in range(self.model_config.num_local_experts):
                for name, _ in (
                    self.model.model.layers[layer_idx].mlp.experts[expert_idx].named_parameters()
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
        """Load non-expert weights (embeddings, attention, norms, lm_head)."""
        logging.info("Loading model skeleton...")

        # Debug: Print skeleton_state_dict keys for verification
        skeleton_keys = list(self.skeleton_state_dict.keys()) if self.skeleton_state_dict else []
        logging.info(f"Skeleton state_dict has {len(skeleton_keys)} keys")
        if skeleton_keys:
            # Print first 20 keys at INFO level to see actual naming
            logging.info(f"Sample skeleton keys (first 20): {skeleton_keys[:20]}")
            # Also print keys that should match embed/norm/lm_head
            embed_keys = [k for k in skeleton_keys if 'embed' in k.lower()][:5]
            norm_keys = [k for k in skeleton_keys if 'norm' in k.lower() and 'layer' not in k.lower()][:5]
            lm_head_keys = [k for k in skeleton_keys if 'lm_head' in k.lower() or 'head' in k.lower()][:5]
            logging.info(f"Embed-related keys: {embed_keys}")
            logging.info(f"Final norm-related keys: {norm_keys}")
            logging.info(f"LM head-related keys: {lm_head_keys}")

        # Debug: Print expected model parameter names (non-expert)
        expected_params = [n for n, _ in self.model.named_parameters() if "experts" not in n]
        logging.info(f"Model expects {len(expected_params)} non-expert parameters")
        logging.debug(f"Sample expected params: {expected_params[:15]}")

        # Filter skeleton state dict for non-expert weights
        for name, param in self.model.named_parameters():
            if "experts" in name:
                continue  # Skip expert weights (loaded separately with MXFP4)

            if name in self.skeleton_state_dict:
                skeleton_tensor = self.skeleton_state_dict[name]
                # Debug: Log shape comparison for first few params and any mismatches
                if param.shape != skeleton_tensor.shape:
                    logging.error(f"SHAPE MISMATCH: {name}")
                    logging.error(f"  Model param shape: {param.shape}")
                    logging.error(f"  Skeleton tensor shape: {skeleton_tensor.shape}")
                    logging.error(f"  Skeleton tensor dtype: {skeleton_tensor.dtype}")
                    logging.error(f"  Skeleton tensor values (first 10): {skeleton_tensor.flatten()[:10]}")
                    raise RuntimeError(f"Shape mismatch for {name}: model={param.shape}, skeleton={skeleton_tensor.shape}")
                param.data.copy_(skeleton_tensor)
            else:
                logging.warning(f"Missing skeleton weight: {name}")

        logging.info("Model skeleton loaded")

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
