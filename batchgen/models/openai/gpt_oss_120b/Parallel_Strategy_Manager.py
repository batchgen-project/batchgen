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

        # EP (Expert Parallelism) state variables
        self.local_routed_experts = []
        self.host_routed_experts = []
        self.experts_per_rank = None
        self.routed_expert_gpu_start_idx = None
        self.routed_expert_gpu_end_idx = None
        self.routed_expert_host_start_idx = None
        self.routed_expert_host_end_idx = None
        self.enable_ep_offloading = False
        self.num_local_expert_per_layer = None

        # Communication handler for EP (set during configure_decoding)
        self.comm = None

    def configure_prefill(self) -> Tuple:
        """Configure model skeleton for prefill phase.

        Returns:
            Tuple of (model, weight_copy_task)
        """
        start_time = time.perf_counter()
        timings = {}

        log_gpu_memory("configure_prefill: START")

        # CRITICAL: Delete previous model to free GPU memory
        # This is essential for decode→prefill transitions (Bug Fix 7)
        if getattr(self, 'model', None) is not None:
            logging.info("Deleting previous model to free GPU memory...")
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
            log_gpu_memory("configure_prefill: After deleting previous model")

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

            # Routed experts (MXFP4 quantized) — use known param names directly
            # instead of named_parameters() to avoid iterating nn.Module objects
            _expert_param_names = [
                "gate_proj.weight", "gate_proj.bias",
                "up_proj.weight", "up_proj.bias",
                "down_proj.weight", "down_proj.bias",
            ]
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
                # Only add to weight_copy_task if offloading is enabled
                # When not offloading, experts are persistent (MXFP4 stored on GPU)
                if experts_need_offloading:
                    self.weight_copy_task["routed_expert"].append(module_key)

        timings["mapping_build"] = time.perf_counter() - step_start

        # Step 5: Load model skeleton and configure modules
        step_start = time.perf_counter()
        self._load_model_skeleton()
        timings["skeleton_load"] = time.perf_counter() - step_start
        log_gpu_memory("configure_prefill: After skeleton load")

        step_start = time.perf_counter()
        self._config_attn_module()
        timings["attn_config"] = time.perf_counter() - step_start
        log_gpu_memory("configure_prefill: After attention config")

        # Step 5.5: Load expert MXFP4 weights if persistent (offloading disabled)
        # Must be called before _config_expert_module() so model attrs exist for registration
        if not experts_need_offloading:
            step_start = time.perf_counter()
            self._load_expert_module()
            timings["expert_load"] = time.perf_counter() - step_start
            log_gpu_memory("configure_prefill: After expert module load")

        step_start = time.perf_counter()
        self._config_expert_module()
        timings["expert_config"] = time.perf_counter() - step_start
        log_gpu_memory("configure_prefill: After expert config")

        # Step 6: Swap to prefill MoE for CuTe dequant + torch.matmul (only when persistent)
        if not experts_need_offloading:
            step_start = time.perf_counter()
            self._swap_to_prefill_moe()
            timings["prefill_moe_swap"] = time.perf_counter() - step_start
            log_gpu_memory("configure_prefill: After prefill MoE swap")

        step_start = time.perf_counter()
        self._config_lm_head_hook()
        timings["lm_head_config"] = time.perf_counter() - step_start

        self.model.eval()
        self.model.to(self.engine_config.Basic_Config.device_torch)

        total_time = time.perf_counter() - start_time
        logging.info(f"Model configuration complete in {total_time:.2f}s")
        logging.info(f"Timings: {timings}")
        log_gpu_memory("configure_prefill: END")

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

    def _load_attn_module(self):
        """Load attention weights from core_engine for decode phase.

        Called when attention is persistent (decode mode).
        Loads weights directly to module parameters.

        Following DeepSeek pattern where attention weights are pre-loaded
        for decode phase instead of being fetched on-demand.
        """
        logging.info("Loading attention module weights...")
        device = self.engine_config.Basic_Config.device_torch

        for layer_idx in range(self.model_config.num_hidden_layers):
            attn_module = self.model.model.layers[layer_idx].self_attn
            module_key = f"attn_{layer_idx}"

            # Get weights from core_engine
            tensors = self.core_engine.get_tensor(module_key)



            # Load projection weights (packed QKV + O)
            if "qkv_proj.weight" in tensors:
                attn_module.qkv_proj.weight.data = tensors["qkv_proj.weight"].to(device)
            if "o_proj.weight" in tensors:
                attn_module.o_proj.weight.data = tensors["o_proj.weight"].to(device)

            # Load biases if present
            if "qkv_proj.bias" in tensors:
                attn_module.qkv_proj.bias.data = tensors["qkv_proj.bias"].to(device)
            if "o_proj.bias" in tensors:
                attn_module.o_proj.bias.data = tensors["o_proj.bias"].to(device)

            # Load sink tokens
            if "sinks" in tensors:
                attn_module.sinks.data = tensors["sinks"].to(device)

            logging.debug(f"Loaded attention weights for layer {layer_idx}")

        logging.info(f"Loaded attention weights for {self.model_config.num_hidden_layers} layers")

    def _load_expert_module(self):
        """Load MXFP4 expert weights to model for decode phase.

        Following DeepSeek pattern: load weights to model attributes ONCE,
        then wrapper caches pointers for fast access during forward.

        For MXFP4 format, we store packed weights and scales as module attributes
        (can't use .weight.data directly since MXFP4 is packed format).
        """
        logging.info("Loading MXFP4 expert module weights...")
        device = self.engine_config.Basic_Config.device_torch

        for layer_idx in range(self.model_config.num_hidden_layers):
            for expert_idx in range(self.model_config.num_local_experts):
                module_key = f"routed_expert_{layer_idx}_{expert_idx}"

                # Get MXFP4 weights from Weights_Storage
                tensors = self.core_engine.get_tensor(module_key)

                expert = self.model.model.layers[layer_idx].mlp.experts[expert_idx]

                # Store MXFP4 packed weights as module attributes
                expert.mxfp4_gate_packed = tensors["gate_proj.weight"].to(device)
                expert.mxfp4_gate_scales = tensors["gate_proj.weight_scales"].to(device)
                expert.mxfp4_up_packed = tensors["up_proj.weight"].to(device)
                expert.mxfp4_up_scales = tensors["up_proj.weight_scales"].to(device)
                expert.mxfp4_down_packed = tensors["down_proj.weight"].to(device)
                expert.mxfp4_down_scales = tensors["down_proj.weight_scales"].to(device)

                # Handle biases if present
                if "gate_proj.bias" in tensors:
                    expert.mxfp4_gate_bias = tensors["gate_proj.bias"].to(device)
                if "up_proj.bias" in tensors:
                    expert.mxfp4_up_bias = tensors["up_proj.bias"].to(device)
                if "down_proj.bias" in tensors:
                    expert.mxfp4_down_bias = tensors["down_proj.bias"].to(device)

        # Sync to ensure all H2D transfers complete
        torch.cuda.synchronize()

        total_experts = self.model_config.num_hidden_layers * self.model_config.num_local_experts
        logging.info(f"Loaded MXFP4 expert weights for all {total_experts} experts")

    def _load_expert_module_ep(self):
        """Load MXFP4 expert weights for EP mode (local experts only).

        Following DeepSeek pattern: only load experts owned by this rank.
        Expert range: [routed_expert_gpu_start_idx, routed_expert_gpu_end_idx)

        For MXFP4 format, we store packed weights and scales as module attributes.
        """
        logging.info(
            f"Loading MXFP4 expert module weights for EP mode. "
            f"Rank {self.rank}: experts [{self.routed_expert_gpu_start_idx}, {self.routed_expert_gpu_end_idx})"
        )
        device = self.engine_config.Basic_Config.device_torch
        loaded_count = 0

        for layer_idx in range(self.model_config.num_hidden_layers):
            # Only load GPU-resident experts for this rank
            for expert_idx in range(self.routed_expert_gpu_start_idx, self.routed_expert_gpu_end_idx):
                module_key = f"routed_expert_{layer_idx}_{expert_idx}"

                # Get MXFP4 weights from Weights_Storage
                tensors = self.core_engine.get_tensor(module_key)

                expert = self.model.model.layers[layer_idx].mlp.experts[expert_idx]

                # Store MXFP4 packed weights as module attributes
                expert.mxfp4_gate_packed = tensors["gate_proj.weight"].to(device)
                expert.mxfp4_gate_scales = tensors["gate_proj.weight_scales"].to(device)
                expert.mxfp4_up_packed = tensors["up_proj.weight"].to(device)
                expert.mxfp4_up_scales = tensors["up_proj.weight_scales"].to(device)
                expert.mxfp4_down_packed = tensors["down_proj.weight"].to(device)
                expert.mxfp4_down_scales = tensors["down_proj.weight_scales"].to(device)

                # Handle biases if present
                if "gate_proj.bias" in tensors:
                    expert.mxfp4_gate_bias = tensors["gate_proj.bias"].to(device)
                if "up_proj.bias" in tensors:
                    expert.mxfp4_up_bias = tensors["up_proj.bias"].to(device)
                if "down_proj.bias" in tensors:
                    expert.mxfp4_down_bias = tensors["down_proj.bias"].to(device)

                loaded_count += 1

        # Sync to ensure all H2D transfers complete
        torch.cuda.synchronize()

        logging.info(
            f"Loaded MXFP4 expert weights for {loaded_count} local experts "
            f"(EP mode, rank {self.rank})"
        )

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

        # Determine expert range based on EP mode
        # EP mode only active when ranges are computed (decode phase)
        ep_enabled = self.world_size > 1 and self.routed_expert_gpu_start_idx is not None
        if ep_enabled:
            # EP mode: only configure local experts
            expert_start = self.routed_expert_gpu_start_idx
            expert_end = self.routed_expert_gpu_end_idx
        else:
            # Single GPU: configure all experts
            expert_start = 0
            expert_end = self.model_config.num_local_experts

        for layer_idx in range(self.model_config.num_hidden_layers):
            for expert_idx in range(expert_start, expert_end):
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

                # Register MXFP4 weight pointers for persistent experts
                # (Weights already loaded to model attrs by _load_expert_module())
                if persistent:
                    wrapped_expert._register_mxfp4_weights()
                    persistent_count += 1
                else:
                    dynamic_count += 1

                self.model.model.layers[layer_idx].mlp.experts[expert_idx] = wrapped_expert

        total_experts = self.model_config.num_hidden_layers * (expert_end - expert_start)
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

    def _setup_decode_moe(self, ep_enabled: bool, weight_format: str):
        """Set up unified GptOssMoEDecode for all layers.

        Creates GptOssMoEDecode per layer, configures persistent/non-persistent
        expert lists, and sets up grouped kernel pointer arrays.

        Args:
            ep_enabled: Whether Expert Parallelism is active (world_size > 1)
            weight_format: "mxfp4" or "bf16"
        """
        from .model import GptOssMoEDecode, _HAS_WGMMA_GROUPED
        from batchgen.moe.mxfp4_grouped_gemm import setup_expert_weight_pointers

        device = self.engine_config.Basic_Config.device_torch

        if ep_enabled:
            expert_start = self.routed_expert_gpu_start_idx
            num_persistent = self.num_local_expert_per_layer
            num_total_per_rank = self.experts_per_rank
            # Non-persistent = host-resident experts on this rank
            persistent_indices = list(range(expert_start, expert_start + num_persistent))
            non_persistent_indices = list(range(
                expert_start + num_persistent,
                expert_start + num_total_per_rank
            ))
        else:
            expert_start = 0
            num_persistent = self.model_config.num_local_experts
            persistent_indices = list(range(num_persistent))
            non_persistent_indices = []

        for layer_idx in range(self.model_config.num_hidden_layers):
            old_moe = self.model.model.layers[layer_idx].mlp

            decode_moe = GptOssMoEDecode(
                self.loaded_model_config,
                ep_enabled=ep_enabled,
                comm=self.comm,
            )
            decode_moe.router = old_moe.router
            decode_moe.total_experts = 128 if ep_enabled else self.model_config.num_local_experts
            decode_moe.expert_start = expert_start
            decode_moe.num_local_experts = num_persistent + len(non_persistent_indices)
            decode_moe.persistent_expert_indices = persistent_indices
            decode_moe.non_persistent_expert_indices = non_persistent_indices
            decode_moe.weight_format = weight_format

            # Transfer expert wrappers (for non-persistent single-expert forward)
            for idx in non_persistent_indices:
                decode_moe.experts[idx] = old_moe.experts[idx]

            # Set up grouped kernel pointer arrays for persistent experts
            if num_persistent > 0 and weight_format == "mxfp4" and _HAS_WGMMA_GROUPED:
                gate_weights, gate_scales = [], []
                up_weights, up_scales = [], []
                down_weights, down_scales = [], []
                gate_biases, up_biases, down_biases = [], [], []
                has_biases = False

                for idx in persistent_indices:
                    wrapper = old_moe.experts[idx]
                    gate_weights.append(wrapper.mxfp4_gate_packed.contiguous())
                    gate_scales.append(wrapper.mxfp4_gate_scales.contiguous())
                    up_weights.append(wrapper.mxfp4_up_packed.contiguous())
                    up_scales.append(wrapper.mxfp4_up_scales.contiguous())
                    down_weights.append(wrapper.mxfp4_down_packed.contiguous())
                    down_scales.append(wrapper.mxfp4_down_scales.contiguous())

                    gb = getattr(wrapper, 'mxfp4_gate_bias', None)
                    if gb is not None:
                        has_biases = True
                        gate_biases.append(gb.contiguous())
                        up_biases.append(getattr(wrapper, 'mxfp4_up_bias').contiguous())
                        down_biases.append(getattr(wrapper, 'mxfp4_down_bias').contiguous())

                decode_moe.gate_ptrs, decode_moe.gate_scale_ptrs = setup_expert_weight_pointers(
                    gate_weights, gate_scales)
                decode_moe.up_ptrs, decode_moe.up_scale_ptrs = setup_expert_weight_pointers(
                    up_weights, up_scales)
                decode_moe.down_ptrs, decode_moe.down_scale_ptrs = setup_expert_weight_pointers(
                    down_weights, down_scales)

                decode_moe.gate_weight_ref = gate_weights[0]
                decode_moe.gate_scale_ref = gate_scales[0]
                decode_moe.down_weight_ref = down_weights[0]
                decode_moe.down_scale_ref = down_scales[0]

                # Keep references alive
                decode_moe._persistent_gate_weights = gate_weights
                decode_moe._persistent_gate_scales = gate_scales
                decode_moe._persistent_up_weights = up_weights
                decode_moe._persistent_up_scales = up_scales
                decode_moe._persistent_down_weights = down_weights
                decode_moe._persistent_down_scales = down_scales

                if has_biases:
                    decode_moe.gate_bias_ptrs = torch.tensor(
                        [b.data_ptr() for b in gate_biases], dtype=torch.int64, device=device)
                    decode_moe.up_bias_ptrs = torch.tensor(
                        [b.data_ptr() for b in up_biases], dtype=torch.int64, device=device)
                    decode_moe.down_bias_ptrs = torch.tensor(
                        [b.data_ptr() for b in down_biases], dtype=torch.int64, device=device)
                    decode_moe._persistent_gate_biases = gate_biases
                    decode_moe._persistent_up_biases = up_biases
                    decode_moe._persistent_down_biases = down_biases

            # Store per-expert weight references for non-persistent single-expert path
            if non_persistent_indices and weight_format == "mxfp4":
                # Build sparse weight lists indexed by global expert idx
                num_total = 128 if ep_enabled else self.model_config.num_local_experts
                decode_moe.gate_weights = [None] * num_total
                decode_moe.gate_scales = [None] * num_total
                decode_moe.up_weights = [None] * num_total
                decode_moe.up_scales = [None] * num_total
                decode_moe.down_weights = [None] * num_total
                decode_moe.down_scales = [None] * num_total
                decode_moe.gate_biases = [None] * num_total
                decode_moe.up_biases = [None] * num_total
                decode_moe.down_biases = [None] * num_total

                for idx in non_persistent_indices:
                    wrapper = old_moe.experts[idx]
                    decode_moe.gate_weights[idx] = wrapper.mxfp4_gate_packed
                    decode_moe.gate_scales[idx] = wrapper.mxfp4_gate_scales
                    decode_moe.up_weights[idx] = wrapper.mxfp4_up_packed
                    decode_moe.up_scales[idx] = wrapper.mxfp4_up_scales
                    decode_moe.down_weights[idx] = wrapper.mxfp4_down_packed
                    decode_moe.down_scales[idx] = wrapper.mxfp4_down_scales
                    decode_moe.gate_biases[idx] = getattr(wrapper, 'mxfp4_gate_bias', None)
                    decode_moe.up_biases[idx] = getattr(wrapper, 'mxfp4_up_bias', None)
                    decode_moe.down_biases[idx] = getattr(wrapper, 'mxfp4_down_bias', None)

            decode_moe.to(device)

            if ep_enabled:
                decode_moe.init_num_tokens(self.padding_bsz)

            self.model.model.layers[layer_idx].mlp = decode_moe

        logging.info(
            f"Set up GptOssMoEDecode for {self.model_config.num_hidden_layers} layers "
            f"(EP={ep_enabled}, format={weight_format}, "
            f"persistent={len(persistent_indices)}, non-persistent={len(non_persistent_indices)})"
        )

    def _swap_to_prefill_moe(self):
        """Swap GptOssMoE to GptOssMoEPrefill.

        Sets up grouped WGMMA for persistent experts and per-expert fallback
        for non-persistent experts. Currently called only when all experts are
        persistent (experts_need_offloading=False).
        """
        from batchgen.models.openai.gpt_oss_120b.model import GptOssMoEPrefill, _HAS_WGMMA_GROUPED
        from batchgen.moe.mxfp4_grouped_gemm import setup_expert_weight_pointers

        device = self.engine_config.Basic_Config.device_torch
        num_experts = self.model_config.num_local_experts

        # All experts are persistent (this method is only called when not offloading)
        persistent_indices = list(range(num_experts))
        non_persistent_indices = []

        for layer_idx in range(self.model_config.num_hidden_layers):
            old_moe = self.model.model.layers[layer_idx].mlp

            prefill_moe = GptOssMoEPrefill(self.model_config)
            prefill_moe.router = old_moe.router
            prefill_moe.to(device)

            prefill_moe.persistent_expert_indices = persistent_indices
            prefill_moe.non_persistent_expert_indices = non_persistent_indices

            # Collect MXFP4 weights from wrapped experts
            gate_weights, gate_scales = [], []
            up_weights, up_scales = [], []
            down_weights, down_scales = [], []
            gate_biases, up_biases, down_biases = [], [], []
            has_biases = False

            for expert_idx in range(num_experts):
                wrapper = old_moe.experts[expert_idx]

                gate_weights.append(wrapper.mxfp4_gate_packed.contiguous())
                gate_scales.append(wrapper.mxfp4_gate_scales.contiguous())
                up_weights.append(wrapper.mxfp4_up_packed.contiguous())
                up_scales.append(wrapper.mxfp4_up_scales.contiguous())
                down_weights.append(wrapper.mxfp4_down_packed.contiguous())
                down_scales.append(wrapper.mxfp4_down_scales.contiguous())

                gb = getattr(wrapper, 'mxfp4_gate_bias', None)
                if gb is not None:
                    has_biases = True
                    gate_biases.append(gb.contiguous())
                    up_biases.append(getattr(wrapper, 'mxfp4_up_bias').contiguous())
                    down_biases.append(getattr(wrapper, 'mxfp4_down_bias').contiguous())

            # Store weight lists (needed for per-expert fallback path)
            prefill_moe.gate_weights = gate_weights
            prefill_moe.gate_scales = gate_scales
            prefill_moe.up_weights = up_weights
            prefill_moe.up_scales = up_scales
            prefill_moe.down_weights = down_weights
            prefill_moe.down_scales = down_scales

            if has_biases:
                prefill_moe.gate_biases = torch.stack(gate_biases)
                prefill_moe.up_biases = torch.stack(up_biases)
                prefill_moe.down_biases = torch.stack(down_biases)

            # Set up grouped WGMMA pointer arrays for persistent experts
            if _HAS_WGMMA_GROUPED:
                prefill_moe.gate_ptrs, prefill_moe.gate_scale_ptrs = setup_expert_weight_pointers(
                    gate_weights, gate_scales)
                prefill_moe.up_ptrs, prefill_moe.up_scale_ptrs = setup_expert_weight_pointers(
                    up_weights, up_scales)
                prefill_moe.down_ptrs, prefill_moe.down_scale_ptrs = setup_expert_weight_pointers(
                    down_weights, down_scales)

                prefill_moe.gate_weight_ref = gate_weights[0]
                prefill_moe.gate_scale_ref = gate_scales[0]
                prefill_moe.down_weight_ref = down_weights[0]
                prefill_moe.down_scale_ref = down_scales[0]

                # Keep references alive so pointer arrays don't dangle
                prefill_moe._persistent_gate_weights = gate_weights
                prefill_moe._persistent_gate_scales = gate_scales
                prefill_moe._persistent_up_weights = up_weights
                prefill_moe._persistent_up_scales = up_scales
                prefill_moe._persistent_down_weights = down_weights
                prefill_moe._persistent_down_scales = down_scales

                if has_biases:
                    prefill_moe.gate_bias_ptrs = torch.tensor(
                        [b.data_ptr() for b in gate_biases], dtype=torch.int64, device=device)
                    prefill_moe.up_bias_ptrs = torch.tensor(
                        [b.data_ptr() for b in up_biases], dtype=torch.int64, device=device)
                    prefill_moe.down_bias_ptrs = torch.tensor(
                        [b.data_ptr() for b in down_biases], dtype=torch.int64, device=device)
                    prefill_moe._persistent_gate_biases = gate_biases
                    prefill_moe._persistent_up_biases = up_biases
                    prefill_moe._persistent_down_biases = down_biases

            self.model.model.layers[layer_idx].mlp = prefill_moe

        grouped_str = " + grouped WGMMA" if _HAS_WGMMA_GROUPED else ""
        logging.info(
            f"Swapped all {self.model_config.num_hidden_layers} layers to GptOssMoEPrefill"
            f" (persistent={len(persistent_indices)}{grouped_str})"
        )

    def _compute_expert_ranges(self):
        """Compute expert range assignment for Expert Parallelism (EP).

        Following DeepSeek pattern (lines 330-387):
        - Each rank owns 128 // world_size experts
        - GPU-resident experts: [gpu_start, gpu_end)
        - Host-resident experts: [host_start, host_end) if offloading enabled

        Sets:
            self.experts_per_rank: Number of experts per rank
            self.routed_expert_gpu_start_idx: First GPU-resident expert index
            self.routed_expert_gpu_end_idx: Last GPU-resident expert index (exclusive)
            self.routed_expert_host_start_idx: First host-resident expert index
            self.routed_expert_host_end_idx: Last host-resident expert index (exclusive)
            self.local_routed_experts: List of GPU-resident expert keys
            self.host_routed_experts: List of host-resident expert keys
        """
        NUM_TOTAL_EXPERTS = 128
        NUM_EXPERT_PER_RANK = NUM_TOTAL_EXPERTS // self.world_size

        # Determine offloading behavior based on deployment scenario
        if self.world_size > 8:
            # Multi-node: all experts persistent (no offloading across nodes)
            NUM_LOCAL_EXPERT_PER_LAYER = NUM_EXPERT_PER_RANK
            self.enable_ep_offloading = False
            logging.info(
                f"Rank {self.rank}: Multi-node mode (world_size={self.world_size}). "
                f"All {NUM_EXPERT_PER_RANK} experts per rank are persistent."
            )
        elif self.engine_config.EP_Config.enable_offloading:
            # Single-node with EP offloading
            offload_ratio = self.engine_config.EP_Config.offloading_ratio
            NUM_LOCAL_EXPERT_PER_LAYER = int(NUM_EXPERT_PER_RANK * (1 - offload_ratio))
            self.enable_ep_offloading = True
            logging.info(
                f"Rank {self.rank}: EP with offloading enabled. "
                f"Experts per rank: {NUM_EXPERT_PER_RANK}, "
                f"Persistent (GPU): {NUM_LOCAL_EXPERT_PER_LAYER}, "
                f"Offloaded (host): {NUM_EXPERT_PER_RANK - NUM_LOCAL_EXPERT_PER_LAYER}"
            )
        else:
            # Single-node without offloading: all experts persistent
            NUM_LOCAL_EXPERT_PER_LAYER = NUM_EXPERT_PER_RANK
            self.enable_ep_offloading = False
            logging.info(
                f"Rank {self.rank}: EP without offloading. "
                f"All {NUM_LOCAL_EXPERT_PER_LAYER} experts persistent per rank."
            )

        # Store for later use
        self.num_local_expert_per_layer = NUM_LOCAL_EXPERT_PER_LAYER
        self.experts_per_rank = NUM_EXPERT_PER_RANK

        # Compute expert ranges for this rank
        self.routed_expert_gpu_start_idx = self.global_rank * NUM_EXPERT_PER_RANK
        self.routed_expert_gpu_end_idx = self.routed_expert_gpu_start_idx + NUM_LOCAL_EXPERT_PER_LAYER
        self.routed_expert_host_start_idx = self.routed_expert_gpu_end_idx
        self.routed_expert_host_end_idx = (self.global_rank + 1) * NUM_EXPERT_PER_RANK

        # Build expert lists for all layers
        self.local_routed_experts = []
        self.host_routed_experts = []

        for layer_idx in range(self.model_config.num_hidden_layers):
            # GPU-resident experts (persistent)
            for expert_idx in range(self.routed_expert_gpu_start_idx, self.routed_expert_gpu_end_idx):
                self.local_routed_experts.append(f"routed_expert_{layer_idx}_{expert_idx}")
            # Host-resident experts (offloaded)
            for expert_idx in range(self.routed_expert_host_start_idx, self.routed_expert_host_end_idx):
                self.host_routed_experts.append(f"routed_expert_{layer_idx}_{expert_idx}")

        logging.info(
            f"Rank {self.rank}: Expert range [{self.routed_expert_gpu_start_idx}, {self.routed_expert_host_end_idx}), "
            f"GPU experts: {len(self.local_routed_experts)}, Host experts: {len(self.host_routed_experts)}"
        )

    def _configure_decoding_sglang(self, padding_bsz=None) -> Tuple:
        """Runtime-peel decode: build a decode-only SGLang ModelRunner (GQA) + wrap.

        gpt-oss single-node TP8/EP8 with dp-attention (matching the standalone
        SGLang launch). The native BatchGen decode model is skipped; BatchGen's
        paged KV (which the adapter wraps) sizes from the remaining HBM in the
        worker's GPU-KV init. The KV manager does not exist yet here, so the
        adapter is injected lazily on the first decode forward
        (SGLangGQADecodeModel._ensure_adapter).
        """
        import os as _os

        from batchgen.runner_adapter.sglang_decode_runner import (
            build_sglang_decode_runner_gqa,
        )
        from batchgen.runner_adapter.sglang_decode_model_gqa import (
            SGLangGQADecodeModel,
        )

        self.loaded_model_config.phase = "decode"
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()

        # 8 GPUs/node on H20 -> derive topology from ranks (gpt-oss is single-node).
        local_world = max(1, torch.cuda.device_count())
        nnodes = max(1, self.world_size // local_world)
        node_rank = self.global_rank // local_world
        mem_frac = float(_os.getenv("BATCHGEN_SGLANG_MEM_FRACTION", "0.85"))
        # dist_init_addr is inert (SGLang adopts BatchGen's already-initialized PG).
        dist_addr = _os.getenv("BATCHGEN_SGLANG_DIST_ADDR", "127.0.0.1:30000")

        runner = build_sglang_decode_runner_gqa(
            loaded_model_config=self.loaded_model_config,
            world_size=self.world_size,
            global_rank=self.global_rank,
            local_rank=self.local_rank,
            dist_init_addr=dist_addr,
            dp_size=self.world_size,
            nnodes=nnodes,
            node_rank=node_rank,
            mem_fraction_static=mem_frac,
        )
        self.model = SGLangGQADecodeModel(runner, self.core_engine)
        self.weight_copy_task = {"attn": [], "routed_expert": []}
        if self.rank == 0:
            logging.info(
                "[DECODE] runtime-peel: SGLang GQA decode runner built "
                f"(gpt-oss, nnodes={nnodes}, node_rank={node_rank}, mem_frac={mem_frac}); "
                "native BatchGen decode model skipped"
            )
        return self.model, self.weight_copy_task

    def configure_decoding(self, padding_bsz=None, comm=None) -> Tuple:
        """Configure model for decoding phase.

        For single GPU (world_size=1):
        - Mode 3 + EP-offloading disabled
        - All modules persistent, grouped GEMM

        For multi-GPU (world_size>1):
        - Expert Parallelism (EP) enabled
        - Each rank owns 128 // world_size experts
        - AllGather → Route → Process local experts → AllReduce

        Following DeepSeek pattern:
        1. Delete prefill model completely to free GPU memory
        2. Clear CUDA cache
        3. Create new model instance with decode config
        4. Re-load skeleton and configure modules

        Args:
            padding_bsz: Maximum batch size per rank (for EP token buffers)
            comm: MPI communicator (for EP communication)

        Returns:
            Tuple of (model, weight_copy_task)
        """
        log_gpu_memory("configure_decoding: START")

        logging.info("Configuring model for decoding phase...")

        # Runtime-peel: when BATCHGEN_RUNTIME=sglang, decode runs through a
        # decode-only SGLang ModelRunner (GQA path) reading BatchGen's KV via
        # BatchGenGQAKVAdapter. The native BatchGen decode model is NOT built
        # (only SGLang's decode weights are resident). self.model becomes a
        # drop-in SGLangGQADecodeModel the worker drives via self.model(...).
        import os as _os
        self._use_sglang_decode = (
            _os.getenv("BATCHGEN_RUNTIME", "native").lower() == "sglang"
        )
        if self._use_sglang_decode:
            return self._configure_decoding_sglang(padding_bsz=padding_bsz)

        # Store comm and padding_bsz for EP communication
        self.comm = comm
        self.padding_bsz = padding_bsz

        # Check if EP is enabled (world_size > 1)
        ep_enabled = self.world_size > 1
        if ep_enabled:
            logging.info(f"Expert Parallelism enabled: world_size={self.world_size}")
            self._compute_expert_ranges()

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
        # For EP: only host-resident experts need dynamic loading
        # For single GPU: all modules persistent
        for layer_idx in range(self.model_config.num_hidden_layers):
            # Attention parameters (BF16, not quantized)
            for name, _ in self.model.model.layers[layer_idx].self_attn.named_parameters():
                tensor_full_name = f"model.layers.{layer_idx}.self_attn.{name}"
                self.state_dict_name_map[tensor_full_name] = {
                    "module_key": f"attn_{layer_idx}",
                    "tensor_key": name,
                }
            # Note: attn NOT added to weight_copy_task -> persistent=True

            # Routed experts (MXFP4 quantized) — use known param names directly
            _expert_param_names = [
                "gate_proj.weight", "gate_proj.bias",
                "up_proj.weight", "up_proj.bias",
                "down_proj.weight", "down_proj.bias",
            ]
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

        # For EP with offloading: add host-resident experts to weight_copy_task
        if ep_enabled and self.enable_ep_offloading:
            self.weight_copy_task["routed_expert"] = self.host_routed_experts.copy()
            logging.info(
                f"EP offloading: {len(self.host_routed_experts)} host-resident experts "
                f"will be loaded dynamically"
            )

        # Step 6: Load skeleton weights
        torch.cuda.empty_cache()
        self._load_model_skeleton()

        # Step 6.5: Load attention weights (persistent for decode)
        # Following DeepSeek pattern where attention weights are pre-loaded
        self._load_attn_module()

        # Step 6.6: Load expert MXFP4 weights
        # For EP: only load local experts (GPU-resident)
        # For single GPU: load all experts
        if ep_enabled:
            self._load_expert_module_ep()
        else:
            self._load_expert_module()

        log_gpu_memory("configure_decoding: After loading skeleton, attention, and experts")

        # Step 7: Configure modules (all persistent for decode)
        self._config_attn_module()
        self._config_expert_module()
        self._config_lm_head_hook()

        # Step 7.5: Set up unified decode MoE
        # Determine MoE routed expert weight format
        pre_dequant = getattr(self.model_config, 'pre_dequantize_weights', False)
        if pre_dequant:
            weight_format = "bf16"
            logging.info(
                "MoE routed expert weights will be pre-dequantized from MXFP4 to BF16 "
                "(--pre-dequantize-weights). Other weights (attention, router, layernorm, lm_head) are unaffected."
            )
        else:
            weight_format = "mxfp4"
            logging.info(
                "MoE routed expert weights kept as MXFP4 (on-the-fly dequant via fused WGMMA kernels)."
            )

        self._setup_decode_moe(ep_enabled, weight_format)

        if weight_format == "bf16":
            self._dequant_experts_to_bf16()

        self.model.eval()
        self.model.to(self.engine_config.Basic_Config.device_torch)

        log_gpu_memory("configure_decoding: END")

        logging.info(
            f"Decoding configuration complete: "
            f"attn_tasks={len(self.weight_copy_task['attn'])}, "
            f"expert_tasks={len(self.weight_copy_task['routed_expert'])} "
            f"(EP={ep_enabled}, offloading={self.enable_ep_offloading if ep_enabled else False})"
        )

        return self.model, self.weight_copy_task

    def get_weight_copy_task(self) -> Dict[str, List[str]]:
        """Return the weight copy task mapping."""
        return self.weight_copy_task

    def get_state_dict_name_map(self) -> Dict[str, Dict[str, str]]:
        """Return the state dict name mapping."""
        return self.state_dict_name_map

    def set_num_tokens_per_rank(self, num_tokens_per_rank: int):
        """Dynamically update num_tokens_per_rank for all MoE layers.

        Called by worker after AllReduce(MAX) to synchronize buffer size
        across all ranks before each decode step.

        Args:
            num_tokens_per_rank: The max batch size across all ranks for this page
        """
        for layer_idx in range(self.model_config.num_hidden_layers):
            layer = self.model.model.layers[layer_idx].mlp
            if hasattr(layer, "set_num_tokens_per_rank"):
                layer.set_num_tokens_per_rank(num_tokens_per_rank)

    def _dequant_experts_to_bf16(self):
        """Pre-dequantize MXFP4 expert weights to BF16 for EP mode.

        Called when world_size >= 4 for better HBM utilization.
        Pre-dequantizing allows direct BF16 GEMM during inference,
        avoiding the overhead of per-forward MXFP4 dequantization.

        Memory impact for 4-way EP (32 experts per rank × 36 layers):
        - MXFP4 per expert: ~35 MB (gate + up + down in MXFP4)
        - BF16 per expert: ~142 MB (gate + up + down in BF16)
        - Total increase per rank: ~4.3 GB (acceptable with 4+ GPUs)
        """
        from batchgen.quantization.mxfp4 import mxfp4_dequantize

        device = self.engine_config.Basic_Config.device_torch
        dequant_count = 0

        logging.info(
            f"Pre-dequantizing MXFP4 expert weights to BF16 for EP mode "
            f"(world_size={self.world_size}, {self.num_local_expert_per_layer} experts per layer)"
        )

        for layer_idx in range(self.model_config.num_hidden_layers):
            for local_e in range(self.num_local_expert_per_layer):
                global_e = self.routed_expert_gpu_start_idx + local_e
                expert_wrapper = self.model.model.layers[layer_idx].mlp.experts[global_e]

                # Dequant gate: [intermediate_size, hidden_size]
                gate_weight_bf16 = mxfp4_dequantize(
                    expert_wrapper.mxfp4_gate_packed,
                    expert_wrapper.mxfp4_gate_scales,
                    dtype=torch.bfloat16
                )
                # Dequant up: [intermediate_size, hidden_size]
                up_weight_bf16 = mxfp4_dequantize(
                    expert_wrapper.mxfp4_up_packed,
                    expert_wrapper.mxfp4_up_scales,
                    dtype=torch.bfloat16
                )
                # Dequant down: [hidden_size, intermediate_size]
                down_weight_bf16 = mxfp4_dequantize(
                    expert_wrapper.mxfp4_down_packed,
                    expert_wrapper.mxfp4_down_scales,
                    dtype=torch.bfloat16
                )

                # Store BF16 weights on the wrapper
                expert_wrapper.gate_weight_bf16 = gate_weight_bf16
                expert_wrapper.up_weight_bf16 = up_weight_bf16
                expert_wrapper.down_weight_bf16 = down_weight_bf16

                # Copy biases if present
                expert_wrapper.gate_bias = expert_wrapper.mxfp4_gate_bias
                expert_wrapper.up_bias = expert_wrapper.mxfp4_up_bias
                expert_wrapper.down_bias = expert_wrapper.mxfp4_down_bias

                # Free MXFP4 weights to reclaim memory
                del expert_wrapper.mxfp4_gate_packed
                del expert_wrapper.mxfp4_gate_scales
                del expert_wrapper.mxfp4_up_packed
                del expert_wrapper.mxfp4_up_scales
                del expert_wrapper.mxfp4_down_packed
                del expert_wrapper.mxfp4_down_scales

                # Mark wrapper as using BF16 weights
                expert_wrapper.use_bf16_weights = True

                dequant_count += 1

        # Force garbage collection and CUDA cache cleanup
        gc.collect()
        torch.cuda.empty_cache()

        logging.info(
            f"Pre-dequantized {dequant_count} experts to BF16 "
            f"({self.num_local_expert_per_layer} experts × {self.model_config.num_hidden_layers} layers)"
        )
