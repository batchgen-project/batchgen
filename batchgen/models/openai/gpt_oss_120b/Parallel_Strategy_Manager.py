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
        """Build state_dict_name_map and weight_copy_task."""
        self.state_dict_name_map = {}
        self.weight_copy_task = {
            "attn": [],
            "routed_expert": [],
        }

        num_layers = self.model_config.num_hidden_layers
        num_experts = self.model_config.num_local_experts

        for layer_idx in range(num_layers):
            # Attention weights (BF16)
            attn_tensors = ["qkv.weight", "qkv.bias", "out.weight", "out.bias"]
            for tensor_name in attn_tensors:
                ckpt_name = f"block.{layer_idx}.attn.{tensor_name}"
                self.state_dict_name_map[ckpt_name] = {
                    "module_key": f"attn_{layer_idx}",
                    "tensor_key": tensor_name,
                }
            self.weight_copy_task["attn"].append(f"attn_{layer_idx}")

            # Expert weights (MXFP4) - sliced format
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
            f"Weight mappings: {len(self.weight_copy_task['attn'])} attn, "
            f"{len(self.weight_copy_task['routed_expert'])} experts"
        )

    def _load_model_skeleton(self):
        """Load skeleton weights (embeddings, norms, router, attention sinks)."""
        if self.skeleton_state_dict is None:
            logging.warning("No skeleton_state_dict provided")
            return

        device = self.engine_config.Basic_Config.device_torch
        loaded = 0
        skipped = 0

        # Direct mapping from checkpoint names to model parameter names
        # Note: Model uses 'block' (singular), RMSNorm uses 'scale' (not 'weight')
        # unembedding has bias=False in model.py, so no unembedding.bias
        mappings = {
            "embedding.weight": "embedding.weight",
            "unembedding.weight": "unembedding.weight",
            "norm.scale": "norm.scale",  # RMSNorm uses .scale
        }

        # Add per-layer mappings
        # Model uses self.block (singular ModuleList), RMSNorm uses .scale
        for layer_idx in range(self.model_config.num_hidden_layers):
            mappings.update({
                # Attention norm and sinks
                f"block.{layer_idx}.attn.norm.scale": f"block.{layer_idx}.attn.norm.scale",
                f"block.{layer_idx}.attn.sinks": f"block.{layer_idx}.attn.sinks",
                # Attention QKV and output projections (BF16, not quantized)
                f"block.{layer_idx}.attn.qkv.weight": f"block.{layer_idx}.attn.qkv.weight",
                f"block.{layer_idx}.attn.qkv.bias": f"block.{layer_idx}.attn.qkv.bias",
                f"block.{layer_idx}.attn.out.weight": f"block.{layer_idx}.attn.out.weight",
                f"block.{layer_idx}.attn.out.bias": f"block.{layer_idx}.attn.out.bias",
                # MLP norm and gate (router)
                f"block.{layer_idx}.mlp.norm.scale": f"block.{layer_idx}.mlp.norm.scale",
                f"block.{layer_idx}.mlp.gate.weight": f"block.{layer_idx}.mlp.gate.weight",
                f"block.{layer_idx}.mlp.gate.bias": f"block.{layer_idx}.mlp.gate.bias",
            })

        # Access transformer through wrapper
        transformer = self.model.transformer if hasattr(self.model, 'transformer') else self.model
        model_state = dict(transformer.named_parameters())

        for ckpt_name, model_name in mappings.items():
            if ckpt_name in self.skeleton_state_dict:
                if model_name in model_state:
                    tensor = self.skeleton_state_dict[ckpt_name]
                    model_state[model_name].data.copy_(tensor.to(device))
                    loaded += 1
                else:
                    skipped += 1
                    logging.debug(f"Model param not found: {model_name}")

        logging.info(f"Skeleton loaded: {loaded} weights, {skipped} skipped")

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
