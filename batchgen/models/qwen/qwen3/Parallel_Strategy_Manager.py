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

"""Parallel Strategy Manager for Qwen3.

Handles model skeleton creation, weight loading, and execution strategy.
Simplified for dense model on single device.
"""

import gc
import logging
import time
from typing import Dict, List, Tuple

import torch

from .model import Qwen3ForCausalLM, Qwen3RotaryEmbedding
from .wrappers import Qwen3AttnWrapper, Qwen3MLPWrapper


class Qwen3ParallelStrategyManager:
    """Manage execution strategy for Qwen3 dense model.

    Single device deployment:
    - Dense model (no MoE)
    - No tensor parallelism
    - BF16 weights (no quantization)
    - All weights persistent on GPU
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

    def configure_prefill(self) -> Tuple:
        """Configure model skeleton for prefill phase.

        Returns:
            Tuple of (model, weight_copy_task)
        """
        start_time = time.perf_counter()

        if getattr(self, 'model', None) is not None:
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()

        # Set phase to prefill for attention wrapper routing
        Qwen3AttnWrapper.phase = "prefill"
        Qwen3MLPWrapper.phase = "prefill"

        # Step 1: Initialize model
        self.model = Qwen3ForCausalLM(self.loaded_model_config)

        # Step 2: Build weight mappings
        # Only attention uses dynamic loading via core engine
        # MLP weights are persistent in skeleton (8B model fits in GPU memory)
        self.state_dict_name_map = {}
        self.weight_copy_task = {}
        self.weight_copy_task["attn"] = []

        for layer_idx in range(self.model_config.num_hidden_layers):
            # Attention weights — dynamic loading via core engine
            for name, _ in self.model.model.layers[layer_idx].self_attn.named_parameters():
                tensor_full_name = f"model.layers.{layer_idx}.self_attn.{name}"
                self.state_dict_name_map[tensor_full_name] = {
                    "module_key": f"attn_{layer_idx}",
                    "tensor_key": name,
                }
            self.weight_copy_task["attn"].append(f"attn_{layer_idx}")
            # MLP weights NOT registered — go to skeleton (persistent)

        # Step 3: Load skeleton weights
        self._load_model_skeleton()

        # Step 4: Create RoPE embedding on model
        self._setup_rope()

        # Step 5: Load attention weights from core engine (persistent mode)
        self._load_attn_module()

        # Step 6: Configure attention wrappers (MLP is persistent in skeleton)
        self._config_attn_module()
        self._config_lm_head_hook()

        self.model.eval()
        self.model.to(self.engine_config.Basic_Config.device_torch)

        total_time = time.perf_counter() - start_time
        logging.info(f"Qwen3 model configuration complete in {total_time:.2f}s")

        return self.model, self.weight_copy_task

    def _load_model_skeleton(self):
        """Load skeleton weights (embeddings, layer norms, lm_head).

        Parameters in state_dict_name_map (attention, MLP) are loaded dynamically.
        """
        if self.skeleton_state_dict is None:
            logging.warning("No skeleton_state_dict provided")
            return

        loaded_count = 0
        for name, param in self.model.named_parameters():
            if name in self.skeleton_state_dict:
                tensor = self.skeleton_state_dict[name]
                if isinstance(tensor, torch.Tensor):
                    param.data = tensor
                    loaded_count += 1

        logging.info(f"Loaded {loaded_count} skeleton parameters")

    def _load_attn_module(self):
        """Load attention weights from core_engine for persistent mode."""
        logging.info("Loading attention module weights...")
        device = self.engine_config.Basic_Config.device_torch

        for layer_idx in range(self.model_config.num_hidden_layers):
            attn_module = self.model.model.layers[layer_idx].self_attn
            module_key = f"attn_{layer_idx}"

            tensors = self.core_engine.get_tensor(module_key)

            for name, tensor in tensors.items():
                # Navigate to the parameter (e.g., "q_proj.weight" -> attn_module.q_proj.weight)
                parts = name.split(".")
                obj = attn_module
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr_name = parts[-1]
                if hasattr(obj, setattr_name):
                    param = getattr(obj, setattr_name)
                    if isinstance(param, torch.nn.Parameter):
                        param.data = tensor.to(device)
                    else:
                        setattr(obj, setattr_name, tensor.to(device))
                else:
                    setattr(obj, setattr_name, tensor.to(device))

            logging.debug(f"Loaded attention weights for layer {layer_idx}")

        logging.info(f"Loaded attention weights for {self.model_config.num_hidden_layers} layers")

    def _setup_rope(self):
        """Create shared RoPE embedding for all attention layers."""
        config = self.loaded_model_config
        device = self.engine_config.Basic_Config.device_torch

        rotary_emb = Qwen3RotaryEmbedding(
            dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            device=device,
        )

        # Share RoPE across all attention layers
        for layer in self.model.model.layers:
            layer.self_attn.rotary_emb = rotary_emb

    def _config_attn_module(self):
        """Wrap attention modules with Qwen3AttnWrapper."""
        for layer_idx in range(self.model_config.num_hidden_layers):
            attn_module = self.model.model.layers[layer_idx].self_attn
            wrapper = Qwen3AttnWrapper(
                module=attn_module,
                layer_idx=layer_idx,
                core_engine=self.core_engine,
                engine_config=self.engine_config,
                model_config=self.model_config,
                persistent=True,
            )
            self.model.model.layers[layer_idx].self_attn = wrapper

    def _config_mlp_module(self):
        """Wrap MLP modules with Qwen3MLPWrapper."""
        for layer_idx in range(self.model_config.num_hidden_layers):
            mlp_module = self.model.model.layers[layer_idx].mlp
            wrapper = Qwen3MLPWrapper(
                module=mlp_module,
                layer_idx=layer_idx,
                core_engine=self.core_engine,
                engine_config=self.engine_config,
                model_config=self.model_config,
                persistent=True,
            )
            self.model.model.layers[layer_idx].mlp = wrapper

    def _config_lm_head_hook(self):
        """Configure LM head hook for logits computation."""
        # LM head is part of skeleton, already loaded
        pass

    def configure_decoding(self, padding_bsz=None, comm=None) -> Tuple:
        """Configure for decode phase.

        For dense single-device model, decode and prefill share the same model.
        Set phase to 'decode' so AttnWrapperBase routes to _forward_decode.
        """
        result = self.configure_prefill()
        # CRITICAL: Set phase to decode for attention wrapper routing
        Qwen3AttnWrapper.phase = "decode"
        Qwen3MLPWrapper.phase = "decode"
        return result

    def get_weight_copy_task(self) -> Dict[str, List[str]]:
        return self.weight_copy_task

    def get_state_dict_name_map(self) -> Dict[str, Dict[str, str]]:
        return self.state_dict_name_map
