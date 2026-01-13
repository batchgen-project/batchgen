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

Uses OpenAI-style model.py and W4A16 fused dequant-GEMM for MXFP4 experts.

Key components:
1. Model skeleton from OpenAI's model.py (Transformer class)
2. MXFP4 expert weights with W4A16 GEMM (weights in 4-bit, activations in BF16)
3. BF16 attention weights (not quantized)
"""

import gc
import logging
import time
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from .model import Transformer, ModelConfig as GptOssModelConfig
from batchgen.moe.fused_mxfp4_gemm import fused_mxfp4_gemm, fused_mxfp4_grouped_gemm
from dataclasses import dataclass


@dataclass
class CausalLMOutputWithPast:
    """Output wrapper to match HuggingFace-style interface expected by BatchGenWorker."""
    logits: torch.Tensor
    past_key_values: Optional[Tuple] = None


class GptOssCausalLMWrapper(nn.Module):
    """Wrapper for GPT-OSS Transformer to match BatchGenWorker interface.

    The OpenAI model.py Transformer has a simple forward(x) signature,
    but BatchGenWorker expects forward(input_ids, attention_mask, use_cache)
    and returns an object with .logits attribute.
    """

    def __init__(self, transformer: Transformer):
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Forward pass with HuggingFace-style interface.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len] (currently unused by model.py)
            position_ids: Position IDs (currently unused by model.py)
            use_cache: Whether to return past key values (not supported yet)

        Returns:
            CausalLMOutputWithPast with logits tensor
        """
        # The OpenAI Transformer just takes input_ids
        # attention_mask handling is done internally via Attn_Wrapper
        logits = self.transformer(input_ids)
        return CausalLMOutputWithPast(logits=logits)


class GptOssExpertWrapper(nn.Module):
    """Expert wrapper for GPT-OSS W4A16 MXFP4 inference.

    Wraps ExpertMLP modules and handles dynamic MXFP4 weight loading
    from BatchGen's core_engine. When called, loads weights from shared
    memory and passes them to the wrapped module's deepgemm_forward().

    This is the GPT-OSS equivalent of BatchGen's Expert_Wrapper, but
    adapted for MXFP4 (uint8 weights) instead of BF16 weights.
    """

    def __init__(
        self,
        expert_module: nn.Module,
        layer_idx: int,
        expert_idx: int,
        core_engine,
        swiglu_limit: float = 7.0,
    ):
        super().__init__()
        self.module = expert_module
        self.layer_idx = layer_idx
        self.expert_idx = expert_idx
        self.core_engine = core_engine
        self.swiglu_limit = swiglu_limit

        # Module key for core_engine weight lookup
        self.expert_weights_idx = f"routed_expert_{layer_idx}_{expert_idx}"

        # Phase for prefetching (0 = prefill, 1 = decode)
        self.phase = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic MXFP4 weight loading.

        Args:
            hidden_states: [batch, hidden_size] BF16

        Returns:
            output: [batch, hidden_size] BF16
        """
        # Load MXFP4 weights from core_engine
        weights_dict = self.core_engine.get_weights(self.expert_weights_idx, self.phase)

        # Call the expert's deepgemm_forward with loaded weights
        output = self.module.deepgemm_forward(hidden_states, weights_dict)

        # Free weights buffer after use
        self.core_engine.free_weights_buffer(self.expert_weights_idx)

        return output

    def set_phase(self, phase: int):
        """Set the execution phase (0=prefill, 1=decode)."""
        self.phase = phase


class GptOssMXFP4ExpertForward:
    """Standalone MXFP4 expert forward pass (for use without wrapper).

    GPT-OSS expert structure:
    - mlp1 = gate_proj || up_proj (concatenated, MXFP4)
    - SwiGLU with clamping
    - mlp2 = down_proj (MXFP4)

    This can be used when weights are already loaded, e.g., in profiling.
    """

    def __init__(
        self,
        mlp1_packed: torch.Tensor,  # [intermediate*2, hidden//2] uint8
        mlp1_scales: torch.Tensor,  # [intermediate*2, hidden//32] uint8
        mlp1_bias: torch.Tensor,    # [intermediate*2] BF16
        mlp2_packed: torch.Tensor,  # [hidden, intermediate//2] uint8
        mlp2_scales: torch.Tensor,  # [hidden, intermediate//32] uint8
        mlp2_bias: torch.Tensor,    # [hidden] BF16
        swiglu_limit: float = 7.0,
    ):
        self.mlp1_packed = mlp1_packed
        self.mlp1_scales = mlp1_scales
        self.mlp1_bias = mlp1_bias
        self.mlp2_packed = mlp2_packed
        self.mlp2_scales = mlp2_scales
        self.mlp2_bias = mlp2_bias
        self.swiglu_limit = swiglu_limit

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass: W4A16 GEMM with SwiGLU activation.

        Args:
            hidden_states: [batch, hidden_size] BF16

        Returns:
            output: [batch, hidden_size] BF16
        """
        # MLP1: W4A16 GEMM
        mlp1_out = fused_mxfp4_gemm(
            hidden_states,
            self.mlp1_packed,
            self.mlp1_scales,
        )
        mlp1_out = mlp1_out + self.mlp1_bias

        # SwiGLU: split, silu, multiply, clamp
        gate, up = mlp1_out.chunk(2, dim=-1)
        hidden = torch.clamp(
            torch.nn.functional.silu(gate) * up,
            min=-self.swiglu_limit,
            max=self.swiglu_limit,
        )

        # MLP2: W4A16 GEMM
        output = fused_mxfp4_gemm(
            hidden,
            self.mlp2_packed,
            self.mlp2_scales,
        )
        output = output + self.mlp2_bias

        return output


class GptOssParallelStrategyManager:
    """Parallel strategy manager for GPT-OSS-120B with W4A16 MXFP4 experts.

    Handles:
    1. Model skeleton creation (OpenAI model.py)
    2. Expert weight loading (MXFP4 packed + scales)
    3. W4A16 fused dequant-GEMM for expert forward pass
    """

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
        self.state_dict_name_map = {}

        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.rank = global_rank

        # Expert forward functions (created during expert config)
        self.expert_forwards: Dict[Tuple[int, int], GptOssMXFP4ExpertForward] = {}

    def configure_prefill(self) -> Tuple:
        """Configure model skeleton for prefill phase.

        Returns:
            Tuple of (model, weight_copy_task)
        """
        start_time = time.perf_counter()
        timings = {}

        # Step 1: Create OpenAI-style model config
        step_start = time.perf_counter()
        gpt_oss_config = GptOssModelConfig(
            num_hidden_layers=self.model_config.num_hidden_layers,
            num_experts=self.model_config.num_local_experts,
            experts_per_token=getattr(self.model_config, 'num_experts_per_tok', 4),
            vocab_size=getattr(self.model_config, 'vocab_size', 201088),
            hidden_size=self.model_config.hidden_size,
            intermediate_size=self.model_config.intermediate_size,
            head_dim=self.model_config.head_dim,
            num_attention_heads=self.model_config.num_attention_heads,
            num_key_value_heads=self.model_config.num_key_value_heads,
            sliding_window=getattr(self.model_config, 'sliding_window', 128),
        )
        timings['config'] = time.perf_counter() - step_start

        # Step 2: Initialize model skeleton (OpenAI Transformer wrapped for BatchGen)
        step_start = time.perf_counter()
        device = self.engine_config.Basic_Config.device_torch
        transformer = Transformer(gpt_oss_config, device=device)
        self.model = GptOssCausalLMWrapper(transformer)
        timings['model_init'] = time.perf_counter() - step_start

        # Step 3: Build weight copy task mappings
        step_start = time.perf_counter()
        self._build_weight_mappings()
        timings['mappings'] = time.perf_counter() - step_start

        # Step 4: Load skeleton weights
        step_start = time.perf_counter()
        self._load_model_skeleton()
        timings['skeleton'] = time.perf_counter() - step_start

        # Step 5: Configure attention modules
        step_start = time.perf_counter()
        self._config_attn_module()
        timings['attn'] = time.perf_counter() - step_start

        # Step 6: Configure expert modules with W4A16 MXFP4
        step_start = time.perf_counter()
        self._config_expert_module()
        timings['expert'] = time.perf_counter() - step_start

        # Step 7: Finalize
        self.model.eval()

        total_time = time.perf_counter() - start_time
        if self.rank == 0:
            logging.info(
                f"[GPT-OSS PREFILL] Configured in {total_time:.2f}s "
                f"(init={timings['model_init']:.1f}s, skeleton={timings['skeleton']:.1f}s)"
            )

        return self.model, self.weight_copy_task

    def _build_weight_mappings(self):
        """Build state_dict_name_map and weight_copy_task.

        NOTE: For GPT-OSS, attention weights are SKELETON (loaded once at init
        via _load_model_skeleton), NOT dynamically loaded. Only expert weights
        are in state_dict_name_map for dynamic loading via core_engine.get_weights().
        """
        self.state_dict_name_map = {}
        self.weight_copy_task = {
            "attn": [],  # Empty - attention is skeleton, not dynamically loaded
            "routed_expert": [],
        }

        num_layers = self.model_config.num_hidden_layers
        num_experts = self.model_config.num_local_experts

        for layer_idx in range(num_layers):
            # NOTE: Attention weights are NOT added here - they are skeleton weights
            # loaded via _load_model_skeleton() from skeleton_state_dict

            # Expert weights (MXFP4) - sliced format, dynamically loaded
            for expert_idx in range(num_experts):
                module_key = f"routed_expert_{layer_idx}_{expert_idx}"

                for mlp_name in ["mlp1", "mlp2"]:
                    for tensor_type in ["packed", "scales", "bias"]:
                        ckpt_name = f"block.{layer_idx}.mlp.experts.{expert_idx}.{mlp_name}.{tensor_type}"
                        self.state_dict_name_map[ckpt_name] = {
                            "module_key": module_key,
                            "tensor_key": f"{mlp_name}.{tensor_type}",
                        }

                self.weight_copy_task["routed_expert"].append(module_key)

        logging.info(
            f"Weight mappings: {len(self.weight_copy_task['attn'])} attn (skeleton), "
            f"{len(self.weight_copy_task['routed_expert'])} experts (dynamic)"
        )

    def _load_model_skeleton(self):
        """Load skeleton weights (embeddings, norms, router, attention sinks).

        Handles potential naming differences between OpenAI checkpoint format
        and our model.py parameter names through a flexible mapping system.
        """
        if self.skeleton_state_dict is None:
            logging.warning("No skeleton_state_dict provided")
            return

        device = self.engine_config.Basic_Config.device_torch
        loaded = 0
        skipped = 0
        missing = []

        # Access transformer through wrapper
        transformer = self.model.transformer if hasattr(self.model, 'transformer') else self.model
        model_state = dict(transformer.named_parameters())

        # Debug: Log ALL skeleton tensor names to understand checkpoint format
        if self.skeleton_state_dict:
            logging.info(f"=== SKELETON STATE DICT DEBUG ===")
            logging.info(f"Total tensors in skeleton_state_dict: {len(self.skeleton_state_dict)}")
            logging.info(f"All tensor names:")
            for k, v in sorted(self.skeleton_state_dict.items()):
                logging.info(f"  {k}: shape={list(v.shape)}, dtype={v.dtype}")
            logging.info(f"=== END SKELETON DEBUG ===")
        else:
            logging.error("skeleton_state_dict is empty or None!")
            return

        # Debug: Log model parameters
        logging.info(f"=== MODEL PARAMETERS DEBUG ===")
        logging.info(f"Total model parameters: {len(model_state)}")
        for k, v in sorted(model_state.items()):
            if not any(x in k for x in ['experts', 'mlp1', 'mlp2']):  # Skip expert params
                logging.info(f"  {k}: shape={list(v.shape)}")
        logging.info(f"=== END MODEL DEBUG ===")

        # Build flexible name mapping: try multiple checkpoint naming conventions
        # OpenAI checkpoint might use different prefixes
        def find_skeleton_tensor(target_name: str) -> tuple:
            """Find a tensor in skeleton_state_dict with flexible naming."""
            # Direct match first
            if target_name in self.skeleton_state_dict:
                return target_name, self.skeleton_state_dict[target_name]

            # Try common variations
            variations = [
                target_name,                                    # exact match
                f"transformer.{target_name}",                   # transformer. prefix
                f"model.{target_name}",                         # model. prefix
                target_name.replace("embedding.", "tok_embeddings."),  # tok_embeddings
                target_name.replace("unembedding.", "output."),        # output for LM head
                target_name.replace("unembedding.", "lm_head."),       # lm_head for LM head
                target_name.replace(".scale", ".weight"),              # .weight instead of .scale
                target_name.replace("norm.scale", "ln.weight"),        # ln instead of norm
                target_name.replace("block.", "layers."),              # layers instead of block
                target_name.replace("block.", "h."),                   # h instead of block (GPT-2 style)
            ]

            for var in variations:
                if var in self.skeleton_state_dict:
                    return var, self.skeleton_state_dict[var]

            return None, None

        # Build the expected mappings (model parameter names)
        expected_params = {
            "embedding.weight": "embedding.weight",
            "unembedding.weight": "unembedding.weight",
            "norm.scale": "norm.scale",
        }

        # Add per-layer mappings
        for layer_idx in range(self.model_config.num_hidden_layers):
            expected_params.update({
                f"block.{layer_idx}.attn.norm.scale": f"block.{layer_idx}.attn.norm.scale",
                f"block.{layer_idx}.attn.sinks": f"block.{layer_idx}.attn.sinks",
                f"block.{layer_idx}.attn.qkv.weight": f"block.{layer_idx}.attn.qkv.weight",
                f"block.{layer_idx}.attn.qkv.bias": f"block.{layer_idx}.attn.qkv.bias",
                f"block.{layer_idx}.attn.out.weight": f"block.{layer_idx}.attn.out.weight",
                f"block.{layer_idx}.attn.out.bias": f"block.{layer_idx}.attn.out.bias",
                f"block.{layer_idx}.mlp.norm.scale": f"block.{layer_idx}.mlp.norm.scale",
                f"block.{layer_idx}.mlp.gate.weight": f"block.{layer_idx}.mlp.gate.weight",
                f"block.{layer_idx}.mlp.gate.bias": f"block.{layer_idx}.mlp.gate.bias",
            })

        # Load each expected parameter
        for target_name, model_name in expected_params.items():
            ckpt_name, tensor = find_skeleton_tensor(target_name)

            if tensor is None:
                missing.append(target_name)
                continue

            if model_name not in model_state:
                logging.debug(f"Model param not found: {model_name}")
                skipped += 1
                continue

            model_param = model_state[model_name]

            # Shape validation
            if tensor.shape != model_param.shape:
                logging.error(
                    f"Shape mismatch for {target_name}:\n"
                    f"  Checkpoint ({ckpt_name}): {list(tensor.shape)}\n"
                    f"  Model ({model_name}): {list(model_param.shape)}"
                )
                continue

            # Copy tensor to model
            try:
                model_param.data.copy_(tensor.to(device))
                loaded += 1
                if ckpt_name != target_name:
                    logging.info(f"Loaded {ckpt_name} -> {model_name} (name variation)")
            except Exception as e:
                logging.error(f"Failed to copy {ckpt_name} -> {model_name}: {e}")

        # Summary
        logging.info(f"Skeleton loading complete:")
        logging.info(f"  Loaded: {loaded}")
        logging.info(f"  Skipped: {skipped}")
        logging.info(f"  Missing: {len(missing)}")
        if missing and len(missing) <= 20:
            logging.warning(f"  Missing tensors: {missing}")

    def _config_attn_module(self):
        """Configure attention modules.

        GPT-OSS attention uses:
        - Combined QKV projection (block.{N}.attn.qkv)
        - GQA (64 Q heads, 8 KV heads)
        - Alternating sliding (128) / full attention
        - Attention sinks
        """
        num_layers = self.model_config.num_hidden_layers
        device = self.engine_config.Basic_Config.device_torch

        # Access transformer through wrapper
        transformer = self.model.transformer if hasattr(self.model, 'transformer') else self.model

        for layer_idx in range(num_layers):
            attn_block = transformer.block[layer_idx].attn

            # Attention weights will be loaded dynamically via core_engine
            # Configure any special attention handling here
            attn_block.layer_idx = layer_idx

        logging.info(f"Configured {num_layers} attention modules")

    def _config_expert_module(self):
        """Configure expert modules with W4A16 MXFP4 forward pass.

        Wraps each ExpertMLP module with GptOssExpertWrapper, which handles
        dynamic MXFP4 weight loading from core_engine during forward pass.
        """
        num_layers = self.model_config.num_hidden_layers
        num_experts = self.model_config.num_local_experts
        swiglu_limit = getattr(self.model_config, 'swiglu_limit', 7.0)

        # Access transformer through wrapper
        transformer = self.model.transformer if hasattr(self.model, 'transformer') else self.model

        # Wrap each expert module with GptOssExpertWrapper
        for layer_idx in range(num_layers):
            mlp_block = transformer.block[layer_idx].mlp

            # Create new ModuleList with wrapped experts
            wrapped_experts = nn.ModuleList()
            for expert_idx in range(num_experts):
                original_expert = mlp_block.experts[expert_idx]
                wrapped = GptOssExpertWrapper(
                    expert_module=original_expert,
                    layer_idx=layer_idx,
                    expert_idx=expert_idx,
                    core_engine=self.core_engine,
                    swiglu_limit=swiglu_limit,
                )
                wrapped_experts.append(wrapped)

            # Replace original experts with wrapped versions
            mlp_block.experts = wrapped_experts

            # Store reference to core_engine on MLPBlock for potential direct use
            mlp_block.core_engine = self.core_engine

        total_experts = num_layers * num_experts
        logging.info(f"Wrapped {total_experts} expert modules with GptOssExpertWrapper (W4A16 MXFP4)")

    def configure_decoding(self) -> Tuple:
        """Configure model for decoding phase.

        Same as prefill for single-GPU deployment.
        """
        return self.configure_prefill()

    def get_weight_copy_task(self) -> Dict[str, List[str]]:
        """Return the weight copy task mapping."""
        return self.weight_copy_task

    def get_state_dict_name_map(self) -> Dict[str, Dict[str, str]]:
        """Return the state dict name mapping."""
        return self.state_dict_name_map
