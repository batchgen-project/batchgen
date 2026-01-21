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
from .wrappers import GptOssExpertWrapper, GptOssAttnWrapper, log_gpu_memory


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

        # Check if experts need offloading (require dynamic buffer loading)
        # When enable_offloading=False, experts are persistent (stored on GPU)
        experts_need_offloading = self.engine_config.EP_Config.enable_offloading

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
                # Only add to weight_copy_task if offloading is enabled
                # When not offloading, experts are persistent (MXFP4 stored on GPU)
                if experts_need_offloading:
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
        """Load skeleton weights (embeddings, layer norms, router, lm_head).

        Skeleton weights are parameters NOT in state_dict_name_map. Parameters in
        state_dict_name_map (attention, experts) are loaded dynamically via wrappers.

        Note: Due to config_torch_module_initializer() in BatchGen, model parameters
        are created with placeholder shape [1] to save memory. We use direct assignment
        (param.data = tensor) instead of copy_() to load the actual weights.
        """
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

        loaded_count = 0
        missing_count = 0
        skipped_module_weights = 0

        # Load only skeleton weights (NOT in state_dict_name_map)
        # Weights in state_dict_name_map are loaded dynamically via wrappers
        for name, param in self.model.named_parameters():
            if "experts" in name:
                continue  # Skip expert weights (loaded separately with MXFP4)

            # Skip weights that will be loaded via module_weights (in state_dict_name_map)
            if name in self.state_dict_name_map:
                skipped_module_weights += 1
                continue

            if name in self.skeleton_state_dict:
                skeleton_tensor = self.skeleton_state_dict[name]
                # Use direct assignment instead of copy_() because model params
                # are created with placeholder shape [1] due to memory optimization
                # in config_torch_module_initializer()
                param.data = skeleton_tensor.to(param.device)
                loaded_count += 1
                logging.debug(f"Loaded skeleton weight: {name} shape={skeleton_tensor.shape}")
            else:
                logging.warning(f"Missing skeleton weight: {name}")
                missing_count += 1

        logging.info(f"Model skeleton loaded: {loaded_count} weights loaded, {missing_count} missing, {skipped_module_weights} skipped (will load via wrappers)")

    def _config_attn_module(self):
        """Configure attention modules with BatchGen wrappers.

        Following DeepSeek pattern:
        - persistent = module_key NOT IN weight_copy_task
        - In prefill: all modules in weight_copy_task → persistent=False → load from buffer
        - In decode: empty weight_copy_task → persistent=True → pre-loaded on GPU
        """
        logging.info("Configuring attention modules...")

        persistent_count = 0
        dynamic_count = 0

        for layer_idx in range(self.model_config.num_hidden_layers):
            attn = self.model.model.layers[layer_idx].self_attn
            module_key = f"attn_{layer_idx}"

            # persistent = NOT IN weight_copy_task (following DeepSeek pattern)
            persistent = module_key not in self.weight_copy_task.get("attn", [])

            if persistent:
                persistent_count += 1
            else:
                dynamic_count += 1

            # Wrap attention for BatchGen execution
            # GptOssAttnWrapper handles GQA and alternating sliding/full attention
            wrapped_attn = GptOssAttnWrapper(
                module=attn,
                layer_idx=layer_idx,
                core_engine=self.core_engine,
                engine_config=self.engine_config,
                model_config=self.model_config,
                persistent=persistent,  # Pass calculated persistent
            )
            self.model.model.layers[layer_idx].self_attn = wrapped_attn

        logging.info(
            f"Configured {self.model_config.num_hidden_layers} attention modules "
            f"({persistent_count} persistent, {dynamic_count} dynamic)"
        )

    def _config_expert_module(self):
        """Configure expert modules with MXFP4 dequantization."""
        logging.info("Configuring expert modules...")

        persistent_count = 0
        dynamic_count = 0

        for layer_idx in range(self.model_config.num_hidden_layers):
            for expert_idx in range(self.model_config.num_local_experts):
                expert = self.model.model.layers[layer_idx].mlp.experts[expert_idx]
                module_key = f"routed_expert_{layer_idx}_{expert_idx}"

                # Determine persistence: True if NOT in weight_copy_task (like DeepSeek)
                # When persistent, MXFP4 weights are stored on GPU; no buffer needed
                persistent = module_key not in self.weight_copy_task.get("routed_expert", [])

                # Wrap expert for BatchGen execution with MXFP4 support
                # GptOssExpertWrapper handles MXFP4 dequantization internally
                wrapped_expert = GptOssExpertWrapper(
                    module=expert,
                    layer_idx=layer_idx,
                    expert_idx=expert_idx,
                    core_engine=self.core_engine,
                    engine_config=self.engine_config,
                    model_config=self.model_config,
                    persistent=persistent,
                )

                # Pre-store MXFP4 weights on GPU for persistent experts
                if persistent:
                    wrapped_expert._pre_store_mxfp4_weights()
                    persistent_count += 1
                else:
                    dynamic_count += 1

                self.model.model.layers[layer_idx].mlp.experts[expert_idx] = wrapped_expert

        total_experts = self.model_config.num_hidden_layers * self.model_config.num_local_experts
        logging.info(
            f"Configured {total_experts} expert modules "
            f"({persistent_count} persistent, {dynamic_count} dynamic)"
        )

    def _config_lm_head_hook(self):
        """Configure LM head for output processing."""
        logging.info("Configuring LM head...")
        # LM head is in BF16, no special handling needed for MXFP4

    def _clear_expert_stored_weights(self):
        """Clear stored MXFP4 weights from all expert wrappers.

        Call this before reconfiguring experts to free GPU memory.
        Prevents OOM during phase transitions (prefill -> decode).
        """
        log_gpu_memory("Before clearing expert weights")

        cleared_count = 0
        for layer_idx in range(self.model_config.num_hidden_layers):
            for expert_idx in range(self.model_config.num_local_experts):
                wrapper = self.model.model.layers[layer_idx].mlp.experts[expert_idx]
                if hasattr(wrapper, '_clear_stored_mxfp4_weights'):
                    wrapper._clear_stored_mxfp4_weights()
                    cleared_count += 1

        # Force garbage collection and CUDA cache cleanup
        gc.collect()
        torch.cuda.empty_cache()

        log_gpu_memory(f"After clearing {cleared_count} expert weights")

    def configure_decoding(self, padding_bsz=None, comm=None) -> Tuple:
        """Configure model for decoding phase (mode 3 + EP-offloading disabled).

        Following DeepSeek pattern:
        1. Delete prefill model completely to free GPU memory
        2. Clear CUDA cache
        3. Create new model instance with decode config
        4. Re-load skeleton and configure modules

        Args:
            padding_bsz: Maximum batch size per rank (unused for single GPU)
            comm: MPI communicator (unused for single GPU)

        Returns:
            Tuple of (model, weight_copy_task)
        """
        log_gpu_memory("configure_decoding: START")

        logging.info("Configuring model for decoding phase...")

        # Step 1: Set phase to decode
        self.loaded_model_config.phase = "decode"

        # Step 2: Delete prefill model completely (DeepSeek pattern)
        # This releases all GPU memory held by the old model and its wrappers
        logging.info("Deleting prefill model to free GPU memory...")
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()

        log_gpu_memory("configure_decoding: After deleting prefill model")

        # Step 3: Reset data structures
        self.state_dict_name_map = {}
        self.weight_copy_task = {}
        self.weight_copy_task["attn"] = []
        self.weight_copy_task["routed_expert"] = []

        # Step 4: Create new model instance with decode config
        logging.info("Creating new model instance for decode phase...")
        self.model = GptOss(self.loaded_model_config)

        log_gpu_memory("configure_decoding: After creating new model")

        # Step 5: Rebuild weight copy task mappings
        # For decode with mode 3 and no EP-offloading: empty weight_copy_task
        # All modules are persistent (weights stay on GPU)
        for layer_idx in range(self.model_config.num_hidden_layers):
            # Attention parameters (BF16, not quantized)
            for name, _ in self.model.model.layers[layer_idx].self_attn.named_parameters():
                tensor_full_name = f"model.layers.{layer_idx}.self_attn.{name}"
                self.state_dict_name_map[tensor_full_name] = {
                    "module_key": f"attn_{layer_idx}",
                    "tensor_key": name,
                }
            # Note: attn NOT added to weight_copy_task -> persistent=True

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
            # Note: routed_expert NOT added to weight_copy_task -> persistent=True

        # Step 6: Load skeleton weights
        torch.cuda.empty_cache()
        self._load_model_skeleton()

        log_gpu_memory("configure_decoding: After loading skeleton")

        # Step 7: Configure modules (all persistent for decode)
        self._config_attn_module()
        self._config_expert_module()
        self._config_lm_head_hook()

        self.model.eval()
        self.model.to(self.engine_config.Basic_Config.device_torch)

        log_gpu_memory("configure_decoding: END")

        logging.info(
            f"Decoding configuration complete: "
            f"attn_tasks={len(self.weight_copy_task['attn'])}, "
            f"expert_tasks={len(self.weight_copy_task['routed_expert'])} "
            f"(all modules persistent)"
        )

        return self.model, self.weight_copy_task

    def get_weight_copy_task(self) -> Dict[str, List[str]]:
        """Return the weight copy task mapping."""
        return self.weight_copy_task

    def get_state_dict_name_map(self) -> Dict[str, Dict[str, str]]:
        """Return the state dict name mapping."""
        return self.state_dict_name_map
