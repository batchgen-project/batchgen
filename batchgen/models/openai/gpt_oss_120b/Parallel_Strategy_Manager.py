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


class GptOssModelShim(nn.Module):
    """Shim to provide HuggingFace-style attribute access for GPT-OSS Transformer.

    BatchGenWorker expects:
    - model.embed_tokens → transformer.embedding
    - model.layers → transformer.block
    - model.norm → transformer.norm
    - model._use_flash_attention_2 (flag)

    This shim provides these mappings.
    """

    def __init__(self, transformer: Transformer):
        super().__init__()
        self._transformer = transformer
        # BatchGenWorker sets this flag
        self._use_flash_attention_2 = False

    @property
    def embed_tokens(self):
        """Map HuggingFace embed_tokens to OpenAI embedding."""
        return self._transformer.embedding

    @property
    def layers(self):
        """Map HuggingFace layers to OpenAI block."""
        return self._transformer.block

    @property
    def norm(self):
        """Map HuggingFace norm to OpenAI norm."""
        return self._transformer.norm


class GptOssCausalLMWrapper(nn.Module):
    """Wrapper for GPT-OSS Transformer to match BatchGenWorker interface.

    The OpenAI model.py Transformer has a simple forward(x) signature,
    but BatchGenWorker expects forward(input_ids, attention_mask, use_cache)
    and returns an object with .logits attribute.

    Also provides HuggingFace-compatible attribute access via .model shim.
    """

    def __init__(self, transformer: Transformer):
        super().__init__()
        self.transformer = transformer
        # Provide HuggingFace-style model.model.* access
        self.model = GptOssModelShim(transformer)

    @property
    def lm_head(self):
        """Map HuggingFace lm_head to OpenAI unembedding.

        BatchGenWorker accesses model.lm_head.weight for the output projection.
        """
        return self.transformer.unembedding

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

    Supports two modes:
    1. get_weights=True (dynamic): Load weights from core_engine per forward,
       then free buffer after use. Used for remote experts in multi-GPU.
    2. get_weights=False (pre-loaded): Use pre-loaded weights stored at init.
       Used for local experts in single-GPU where all 128 experts fit.

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
        get_weights: bool = True,
        preloaded_weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.module = expert_module
        self.layer_idx = layer_idx
        self.expert_idx = expert_idx
        self.core_engine = core_engine
        self.swiglu_limit = swiglu_limit
        self.get_weights = get_weights

        # Module key for core_engine weight lookup
        self.expert_weights_idx = f"routed_expert_{layer_idx}_{expert_idx}"

        # Phase for prefetching: "prefill" or "decode"
        # core_engine.get_weights() expects (str, str)
        self.phase = "prefill"

        # Pre-loaded weights (for local experts with get_weights=False)
        self.preloaded_weights = preloaded_weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with MXFP4 weights.

        If get_weights=True: Load from core_engine, forward, free buffer.
        If get_weights=False: Use pre-loaded weights directly.

        Args:
            hidden_states: [batch, hidden_size] BF16

        Returns:
            output: [batch, hidden_size] BF16
        """
        if self.get_weights:
            # Dynamic loading mode: load -> forward -> free
            weights_dict = self.core_engine.get_weights(self.expert_weights_idx, self.phase)
            output = self.module.deepgemm_forward(hidden_states, weights_dict)
            self.core_engine.free_weights_buffer(self.expert_weights_idx)
        else:
            # Pre-loaded mode: use cached weights directly
            if self.preloaded_weights is None:
                raise RuntimeError(
                    f"Expert {self.expert_weights_idx} has get_weights=False but no preloaded_weights"
                )
            output = self.module.deepgemm_forward(hidden_states, self.preloaded_weights)

        return output

    def set_phase(self, phase: str):
        """Set the execution phase ("prefill" or "decode")."""
        self.phase = phase


class GptOssAttnWrapper(nn.Module):
    """Attention wrapper for GPT-OSS with HtoD weight fetching.

    Supports two modes:
    1. get_weights=True (HtoD): Load weights from core_engine per forward,
       then free buffer after use.
    2. get_weights=False (skeleton): Weights already loaded in module parameters.

    Following BatchGen's Attn_Wrapper pattern from DeepSeek implementation.
    """
    phase = "prefill"  # Class variable for phase tracking

    def __init__(
        self,
        module: nn.Module,
        layer_idx: int,
        core_engine,
        engine_config,
        model_config,
        get_weights: bool = True,
    ):
        super().__init__()
        self.module = module
        self.layer_idx = layer_idx
        self.core_engine = core_engine
        self.engine_config = engine_config
        self.model_config = model_config
        self.get_weights = get_weights
        self.attn_module_id = f"attn_{layer_idx}"

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass with HtoD weight fetching.

        If get_weights=True: Load weights from core_engine, forward, free buffer.
        If get_weights=False: Use existing weights in module parameters.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size] or [batch*seq_len, hidden_size]
            **kwargs: Additional arguments passed to attention module

        Returns:
            output: Attention output tensor
        """
        if self.get_weights:
            # HtoD: Load attention weights from host
            weights_dict = self.core_engine.get_weights(
                self.attn_module_id, GptOssAttnWrapper.phase
            )

            # Apply weights to module parameters
            # Map tensor_key to module parameter names
            param_mapping = {
                "qkv.weight": "qkv.weight",
                "qkv.bias": "qkv.bias",
                "out.weight": "out.weight",
                "out.bias": "out.bias",
                "norm.scale": "norm.scale",
                "sinks": "sinks",
            }

            for tensor_key, param_name in param_mapping.items():
                if tensor_key in weights_dict:
                    # Navigate to the parameter
                    parts = param_name.split('.')
                    target = self.module
                    for part in parts[:-1]:
                        target = getattr(target, part)
                    setattr(target, parts[-1], nn.Parameter(
                        weights_dict[tensor_key], requires_grad=False
                    ))

        # Execute attention forward
        output = self.module(hidden_states, **kwargs)

        if self.get_weights:
            # Free GPU buffer
            self.core_engine.free_weights_buffer(self.attn_module_id)

        return output

    def _unregister_fp8_weights(self):
        """No-op for FP8 weight unregistration.

        GPT-OSS uses MXFP4 (not FP8), so this is a no-op.
        BatchGenWorker calls this after prefill cleanup.
        """
        pass

    @classmethod
    def set_phase(cls, phase: str):
        """Set the execution phase for all instances."""
        cls.phase = phase


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

        # Early validation of skeleton_state_dict
        if self.skeleton_state_dict is None:
            raise RuntimeError(
                "skeleton_state_dict is None! This means the parameter server didn't return "
                "skeleton weights. Check that ps.parameter_server.get_skeleton_state_dict() "
                "is returning the expected tensors."
            )
        if len(self.skeleton_state_dict) == 0:
            raise RuntimeError(
                "skeleton_state_dict is empty! This means no tensors were loaded as skeleton weights. "
                "All tensors may have been added to state_dict_name_map instead. "
                "Check the _parse_state_dict() logic in gpt_oss_parameter_server.py."
            )
        logging.info(f"skeleton_state_dict received with {len(self.skeleton_state_dict)} tensors")
        # Log first few tensor names as early indicator of naming convention
        sample_keys = list(self.skeleton_state_dict.keys())[:5]
        logging.info(f"Sample skeleton tensor names: {sample_keys}")

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

        # Step 6: Pre-load local expert weights (for world_size==1)
        step_start = time.perf_counter()
        self._load_local_routed_experts()
        timings['expert_preload'] = time.perf_counter() - step_start

        # Step 7: Configure expert modules with W4A16 MXFP4
        step_start = time.perf_counter()
        self._config_expert_module()
        timings['expert'] = time.perf_counter() - step_start

        # Step 8: Finalize
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

        Attention weights are SKELETON (loaded once at init, ~3GB).
        Expert weights use HtoD (dynamic loading, ~55GB).

        For world_size==1 (single GPU), ALL expert weights are also loaded as
        skeleton (pre-loaded at init) to avoid HtoD buffer contention issues.
        The original circular queue eviction logic works well when
        buffer_slots >= local_experts, but GPT-OSS has 128 experts with only
        8 buffer slots.

        For world_size>1, expert weights are loaded dynamically via HtoD
        (fewer local experts per GPU makes circular queue viable).
        """
        self.state_dict_name_map = {}
        self.weight_copy_task = {
            "routed_expert": [],
        }

        num_layers = self.model_config.num_hidden_layers
        num_experts = self.model_config.num_local_experts

        # For world_size==1, all experts are local and pre-loaded (skeleton mode)
        # For world_size>1, experts are loaded dynamically via HtoD
        all_experts_local = (self.world_size == 1)

        # Build local_routed_experts list for pre-loading
        self.local_routed_experts = []

        for layer_idx in range(num_layers):
            # =================================================================
            # Attention weights (BF16) - SKELETON (loaded from skeleton_state_dict)
            # NOT in state_dict_name_map, loaded by _load_model_skeleton()
            # =================================================================
            # (no state_dict_name_map entries for attention)

            # =================================================================
            # Expert weights (MXFP4) - HtoD via GptOssExpertWrapper
            # =================================================================
            for expert_idx in range(num_experts):
                module_key = f"routed_expert_{layer_idx}_{expert_idx}"

                for mlp_name in ["mlp1", "mlp2"]:
                    for tensor_type in ["packed", "scales", "bias"]:
                        ckpt_name = f"block.{layer_idx}.mlp.experts.{expert_idx}.{mlp_name}.{tensor_type}"
                        self.state_dict_name_map[ckpt_name] = {
                            "module_key": module_key,
                            "tensor_key": f"{mlp_name}.{tensor_type}",
                        }

                if all_experts_local:
                    # Pre-load at init (skeleton mode) - don't add to weight_copy_task
                    self.local_routed_experts.append(module_key)
                else:
                    # Dynamic loading via HtoD
                    self.weight_copy_task["routed_expert"].append(module_key)

        if all_experts_local:
            logging.info(
                f"Weight mappings (world_size=1): "
                f"attention (skeleton), "
                f"{len(self.local_routed_experts)} experts (pre-loaded)"
            )
        else:
            logging.info(
                f"Weight mappings: "
                f"attention (skeleton), "
                f"{len(self.weight_copy_task['routed_expert'])} experts (HtoD)"
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

        # Build flexible name mapping: try multiple checkpoint naming conventions
        # OpenAI checkpoint might use different prefixes
        def find_skeleton_tensor(target_name: str) -> tuple:
            """Find a tensor in skeleton_state_dict with flexible naming.

            GPT-OSS checkpoint may use various naming conventions:
            - block.{N}.attn.qkv.weight (our model.py style)
            - block.{N}.attn_qkv.weight (underscore variant)
            - block.{N}.self_attn.qkv_proj.weight
            - block.{N}.attn.c_attn.weight (GPT-2 style)
            - h.{N}.attn.c_attn.weight (GPT-2 with 'h' prefix)
            """
            # Direct match first
            if target_name in self.skeleton_state_dict:
                return target_name, self.skeleton_state_dict[target_name]

            # Build comprehensive variations
            variations = [target_name]  # exact match first

            # Extract layer index if present (e.g., "block.5.attn.qkv.weight" -> layer=5)
            import re
            layer_match = re.search(r'block\.(\d+)\.', target_name)
            layer_idx = int(layer_match.group(1)) if layer_match else None

            # Common prefix variations
            for prefix in ["", "transformer.", "model.", "transformer.model."]:
                variations.append(f"{prefix}{target_name}")

            # GPT-2 style: block -> h
            variations.append(target_name.replace("block.", "h."))
            variations.append(f"transformer.{target_name.replace('block.', 'h.')}")

            # Attention naming variations
            if ".attn." in target_name:
                # attn -> self_attn
                variations.append(target_name.replace(".attn.", ".self_attn."))
                # attn -> attention
                variations.append(target_name.replace(".attn.", ".attention."))

                # QKV variations
                if ".qkv." in target_name:
                    # qkv -> c_attn (GPT-2)
                    variations.append(target_name.replace(".qkv.", ".c_attn."))
                    # qkv -> qkv_proj
                    variations.append(target_name.replace(".qkv.", ".qkv_proj."))
                    # attn.qkv -> attn_qkv (underscore)
                    variations.append(target_name.replace(".attn.qkv.", ".attn_qkv."))
                    # attn.qkv -> Wqkv
                    variations.append(target_name.replace(".attn.qkv.", ".attn.Wqkv."))

                # Output projection variations
                if ".out." in target_name:
                    # out -> c_proj (GPT-2)
                    variations.append(target_name.replace(".out.", ".c_proj."))
                    # out -> o_proj
                    variations.append(target_name.replace(".out.", ".o_proj."))
                    # out -> out_proj
                    variations.append(target_name.replace(".out.", ".out_proj."))
                    # attn.out -> attn_out (underscore)
                    variations.append(target_name.replace(".attn.out.", ".attn_out."))

                # Norm variations
                if ".norm." in target_name:
                    # norm -> ln (layer norm)
                    variations.append(target_name.replace(".norm.", ".ln."))
                    # norm -> input_layernorm
                    variations.append(target_name.replace(".attn.norm.", ".input_layernorm."))
                    # attn.norm -> attn_norm
                    variations.append(target_name.replace(".attn.norm.", ".attn_norm."))

            # MLP/FFN variations
            if ".mlp." in target_name:
                if ".mlp.norm." in target_name:
                    # mlp.norm -> post_attention_layernorm
                    variations.append(target_name.replace(".mlp.norm.", ".post_attention_layernorm."))
                    # mlp.norm -> mlp_norm
                    variations.append(target_name.replace(".mlp.norm.", ".mlp_norm."))
                if ".mlp.gate." in target_name:
                    # gate -> router
                    variations.append(target_name.replace(".mlp.gate.", ".mlp.router."))
                    # mlp.gate -> moe_gate
                    variations.append(target_name.replace(".mlp.gate.", ".moe_gate."))

            # Embedding variations (GPT-OSS checkpoint may use different names)
            if "embedding." in target_name:
                variations.append(target_name.replace("embedding.", "tok_embeddings."))
                variations.append(target_name.replace("embedding.", "embed_tokens."))
                variations.append(target_name.replace("embedding.", "wte."))
                # Model-prefixed variations
                variations.append(f"model.{target_name}")
                variations.append(target_name.replace("embedding.", "model.embed_tokens."))
                variations.append(target_name.replace("embedding.", "transformer.wte."))

            # Unembedding/LM head variations
            if "unembedding." in target_name:
                variations.append(target_name.replace("unembedding.", "output."))
                variations.append(target_name.replace("unembedding.", "lm_head."))
                # Model-prefixed variations
                variations.append(f"model.{target_name}")
                variations.append(target_name.replace("unembedding.", "model.lm_head."))
                variations.append(target_name.replace("unembedding.", "transformer.wte."))

            # Scale -> weight variations
            if ".scale" in target_name:
                variations.append(target_name.replace(".scale", ".weight"))

            # Add transformer prefix to all variations
            for var in list(variations):
                if not var.startswith("transformer."):
                    variations.append(f"transformer.{var}")

            # Try all variations
            for var in variations:
                if var in self.skeleton_state_dict:
                    return var, self.skeleton_state_dict[var]

            # Fallback: fuzzy match by layer index and tensor type
            # This handles cases where checkpoint naming is completely different
            if layer_idx is not None:
                layer_str = str(layer_idx)
                # Determine what type of tensor we're looking for
                tensor_type = None
                if ".qkv." in target_name and ".weight" in target_name:
                    tensor_type = "qkv_weight"
                    keywords = ["qkv", "c_attn", "q_proj", "wqkv", "in_proj"]
                elif ".qkv." in target_name and ".bias" in target_name:
                    tensor_type = "qkv_bias"
                    keywords = ["qkv", "c_attn", "q_proj", "wqkv", "in_proj"]
                elif ".out." in target_name and ".weight" in target_name:
                    tensor_type = "out_weight"
                    keywords = ["out", "c_proj", "o_proj", "out_proj", "dense"]
                elif ".out." in target_name and ".bias" in target_name:
                    tensor_type = "out_bias"
                    keywords = ["out", "c_proj", "o_proj", "out_proj", "dense"]
                elif ".norm." in target_name:
                    tensor_type = "norm"
                    keywords = ["norm", "ln", "layernorm"]
                elif ".gate." in target_name:
                    tensor_type = "gate"
                    keywords = ["gate", "router"]

                if tensor_type and keywords:
                    is_weight = "weight" in target_name
                    is_bias = "bias" in target_name
                    is_scale = "scale" in target_name

                    for ckpt_name, ckpt_tensor in self.skeleton_state_dict.items():
                        ckpt_lower = ckpt_name.lower()
                        # Must contain layer index
                        if layer_str not in ckpt_name:
                            continue
                        # Must match weight/bias/scale type
                        if is_weight and "weight" not in ckpt_lower:
                            continue
                        if is_bias and "bias" not in ckpt_lower:
                            continue
                        if is_scale and "scale" not in ckpt_lower and "weight" not in ckpt_lower:
                            continue
                        # Check if any keyword matches
                        if any(kw in ckpt_lower for kw in keywords):
                            logging.info(f"Fuzzy matched: {target_name} -> {ckpt_name}")
                            return ckpt_name, ckpt_tensor

            return None, None

        # Build the expected mappings (model parameter names)
        # Attention weights are SKELETON (loaded once at init, ~3GB total)
        # Expert weights use HtoD (dynamic loading, ~55GB total)
        expected_params = {
            "embedding.weight": "embedding.weight",
            "unembedding.weight": "unembedding.weight",
            "norm.scale": "norm.scale",
        }

        # Add per-layer mappings (attention and MLP skeleton)
        for layer_idx in range(self.model_config.num_hidden_layers):
            expected_params.update({
                # Attention weights - skeleton (loaded once at init)
                f"block.{layer_idx}.attn.norm.scale": f"block.{layer_idx}.attn.norm.scale",
                f"block.{layer_idx}.attn.sinks": f"block.{layer_idx}.attn.sinks",
                f"block.{layer_idx}.attn.qkv.weight": f"block.{layer_idx}.attn.qkv.weight",
                f"block.{layer_idx}.attn.qkv.bias": f"block.{layer_idx}.attn.qkv.bias",
                f"block.{layer_idx}.attn.out.weight": f"block.{layer_idx}.attn.out.weight",
                f"block.{layer_idx}.attn.out.bias": f"block.{layer_idx}.attn.out.bias",
                # MLP gate and norm - skeleton (loaded once at init)
                f"block.{layer_idx}.mlp.norm.scale": f"block.{layer_idx}.mlp.norm.scale",
                f"block.{layer_idx}.mlp.gate.weight": f"block.{layer_idx}.mlp.gate.weight",
                f"block.{layer_idx}.mlp.gate.bias": f"block.{layer_idx}.mlp.gate.bias",
            })

        # Helper to find similar tensor names for debugging
        def find_similar_names(target_name: str, skeleton_keys: list, max_results: int = 3) -> list:
            """Find skeleton keys that might match the target name."""
            import re
            # Extract key parts from target name
            parts = target_name.lower().split('.')
            similar = []
            for key in skeleton_keys:
                key_lower = key.lower()
                # Check how many parts match
                matches = sum(1 for p in parts if p in key_lower)
                if matches >= 2:  # At least 2 parts match
                    similar.append((key, matches))
            # Sort by number of matches (descending)
            similar.sort(key=lambda x: x[1], reverse=True)
            return [s[0] for s in similar[:max_results]]

        # Load each expected parameter
        skeleton_keys = list(self.skeleton_state_dict.keys()) if self.skeleton_state_dict else []

        for target_name, model_name in expected_params.items():
            ckpt_name, tensor = find_skeleton_tensor(target_name)

            if tensor is None:
                missing.append(target_name)
                # For critical attention weights, log potential matches
                if "qkv" in target_name or "out" in target_name:
                    similar = find_similar_names(target_name, skeleton_keys)
                    if similar:
                        logging.warning(
                            f"Missing {target_name}, possible matches in skeleton: {similar}"
                        )
                continue

            if model_name not in model_state:
                logging.debug(f"Model param not found: {model_name}")
                skipped += 1
                continue

            model_param = model_state[model_name]

            # Handle placeholder parameters (shape [1] from config_torch_module_initializer)
            # When using memory-efficient init, all params start as [1] placeholders
            # We need to replace the parameter entirely using setattr on the parent module
            try:
                is_placeholder = (model_param.numel() == 1 and tensor.numel() > 1)

                if is_placeholder:
                    # For placeholder replacement, we need to find the parent module
                    # and replace the parameter attribute directly
                    # model_name is like "block.0.attn.qkv.weight"
                    parts = model_name.rsplit('.', 1)
                    if len(parts) == 2:
                        module_path, param_name = parts
                        # Navigate to the parent module
                        parent_module = transformer
                        for attr in module_path.split('.'):
                            if attr.isdigit():
                                parent_module = parent_module[int(attr)]
                            else:
                                parent_module = getattr(parent_module, attr)
                        # Replace the parameter with a new nn.Parameter
                        new_param = nn.Parameter(tensor.to(device), requires_grad=False)
                        setattr(parent_module, param_name, new_param)
                    else:
                        # Top-level parameter (rare case)
                        model_param.data = tensor.to(device)
                        logging.debug(f"Replaced top-level placeholder {model_name}")
                elif tensor.shape == model_param.shape:
                    # Normal copy for matching shapes
                    model_param.data.copy_(tensor.to(device))
                else:
                    # Shape mismatch (not a placeholder case)
                    logging.error(
                        f"Shape mismatch for {target_name}:\n"
                        f"  Checkpoint ({ckpt_name}): {list(tensor.shape)}\n"
                        f"  Model ({model_name}): {list(model_param.shape)}"
                    )
                    continue

                loaded += 1
                if ckpt_name != target_name:
                    logging.info(f"Loaded {ckpt_name} -> {model_name} (name variation)")
            except Exception as e:
                logging.error(f"Failed to load {ckpt_name} -> {model_name}: {e}")
                import traceback
                logging.error(traceback.format_exc())

        # Summary
        logging.info(f"Skeleton loading: {loaded} loaded, {skipped} skipped, {len(missing)} missing")

        # Validate all Linear layers have proper 2D weights
        self._validate_linear_weights(transformer)

    def _validate_linear_weights(self, transformer):
        """Validate Linear and Embedding layers have proper weights.

        Checks that skeleton loading successfully replaced all placeholder weights.
        Raises RuntimeError if critical weights (attention, embedding) are invalid.
        """
        # Validate embedding layers (critical for inference)
        for name, module in transformer.named_modules():
            if isinstance(module, nn.Embedding):
                if module.weight.numel() == 1:
                    raise RuntimeError(
                        f"FATAL: Embedding '{name}' weight is still a placeholder!\n"
                        f"Expected shape: [{module.num_embeddings}, {module.embedding_dim}]\n"
                        f"The checkpoint may use different tensor names for embedding."
                    )

        # Validate Linear layers
        issues = []
        valid_count = 0

        for name, module in transformer.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight
                if weight.numel() == 1:
                    issues.append(f"{name}.weight: placeholder")
                elif weight.dim() != 2:
                    issues.append(f"{name}.weight: {weight.dim()}D (expected 2D)")
                else:
                    valid_count += 1

        if issues:
            # Check if attention weights are still placeholders - this is fatal
            attn_issues = [i for i in issues if 'attn' in i]
            if attn_issues:
                raise RuntimeError(
                    f"FATAL: {len(attn_issues)} attention weights invalid!\n"
                    f"Issues: {attn_issues[:5]}\n"
                    f"Skeleton loading failed. Check checkpoint tensor names."
                )
            logging.warning(f"Weight validation: {len(issues)} issues ({valid_count} OK)")
        else:
            logging.info(f"Weight validation: {valid_count} Linear layers OK")

    def _load_local_routed_experts(self):
        """Pre-load local routed expert weights to GPU.

        For world_size==1, all 128 experts per layer are local and need to be
        pre-loaded at init. This avoids HtoD buffer contention issues where
        the circular queue eviction logic doesn't work well with
        128 experts and only 8 buffer slots.

        Uses core_engine.get_tensor() to load weights directly from storage,
        bypassing the GPU buffer system.
        """
        if not hasattr(self, 'local_routed_experts') or not self.local_routed_experts:
            logging.info("No local routed experts to pre-load (using dynamic HtoD loading)")
            return

        device = self.engine_config.Basic_Config.device_torch
        transformer = self.model.transformer if hasattr(self.model, 'transformer') else self.model

        logging.info(f"Pre-loading {len(self.local_routed_experts)} local routed expert weights to GPU...")

        # Store pre-loaded weights in a dict for wrapper access
        self.preloaded_expert_weights = {}

        for module_key in self.local_routed_experts:
            # Parse layer_idx and expert_idx from module_key
            # Format: "routed_expert_{layer_idx}_{expert_idx}"
            parts = module_key.split("_")
            layer_idx = int(parts[2])
            expert_idx = int(parts[3])

            # Get weights from storage using get_tensor()
            # This returns a dict with keys like 'mlp1.packed', 'mlp1.scales', etc.
            try:
                tensors = self.core_engine.get_tensor(module_key)

                # Move weights to GPU
                weights_gpu = {}
                for key, tensor in tensors.items():
                    weights_gpu[key] = tensor.to(device)

                # Store for wrapper access
                self.preloaded_expert_weights[module_key] = weights_gpu

            except Exception as e:
                logging.error(f"Failed to pre-load expert {module_key}: {e}")
                raise

        logging.info(f"Pre-loaded {len(self.preloaded_expert_weights)} expert weight sets to GPU")

        # Log memory usage
        used_memory = torch.cuda.memory_allocated(device)
        logging.info(f"GPU memory after expert pre-load: {used_memory / (1024**3):.2f} GB")

    def _config_attn_module(self):
        """Configure attention modules with GptOssAttnWrapper.

        GPT-OSS attention uses:
        - Combined QKV projection (block.{N}.attn.qkv)
        - GQA (64 Q heads, 8 KV heads)
        - Alternating sliding (128) / full attention
        - Attention sinks

        Wraps each attention module with GptOssAttnWrapper for HtoD weight fetching.
        """
        num_layers = self.model_config.num_hidden_layers
        device = self.engine_config.Basic_Config.device_torch

        # Access transformer through wrapper
        transformer = self.model.transformer if hasattr(self.model, 'transformer') else self.model

        htod_count = 0
        skeleton_count = 0

        for layer_idx in range(num_layers):
            attn_module = transformer.block[layer_idx].attn

            # Check if attention needs HtoD loading (in weight_copy_task)
            attn_module_key = f"attn_{layer_idx}"
            get_weights = attn_module_key in self.weight_copy_task.get("attn", [])

            # Wrap with GptOssAttnWrapper
            wrapped = GptOssAttnWrapper(
                module=attn_module,
                layer_idx=layer_idx,
                core_engine=self.core_engine,
                engine_config=self.engine_config,
                model_config=self.model_config,
                get_weights=get_weights,
            )

            # Replace attention module with wrapped version
            transformer.block[layer_idx].attn = wrapped

            if get_weights:
                htod_count += 1
            else:
                skeleton_count += 1

        logging.info(
            f"Configured {num_layers} attention modules: "
            f"{htod_count} HtoD, {skeleton_count} skeleton"
        )

    def _config_expert_module(self):
        """Configure expert modules with W4A16 MXFP4 forward pass.

        For local experts (pre-loaded at init), wraps with get_weights=False
        and passes pre-loaded weights. For remote experts, wraps with
        get_weights=True for dynamic loading via HtoD.
        """
        num_layers = self.model_config.num_hidden_layers
        num_experts = self.model_config.num_local_experts
        swiglu_limit = getattr(self.model_config, 'swiglu_limit', 7.0)

        # Check if we have pre-loaded experts
        has_preloaded = hasattr(self, 'preloaded_expert_weights') and self.preloaded_expert_weights

        # Access transformer through wrapper
        transformer = self.model.transformer if hasattr(self.model, 'transformer') else self.model

        local_count = 0
        dynamic_count = 0

        # Wrap each expert module with GptOssExpertWrapper
        for layer_idx in range(num_layers):
            mlp_block = transformer.block[layer_idx].mlp

            # Create new ModuleList with wrapped experts
            wrapped_experts = nn.ModuleList()
            for expert_idx in range(num_experts):
                module_key = f"routed_expert_{layer_idx}_{expert_idx}"
                original_expert = mlp_block.experts[expert_idx]

                # Check if this expert is pre-loaded (local)
                if has_preloaded and module_key in self.preloaded_expert_weights:
                    # Local expert: use pre-loaded weights, no dynamic loading
                    wrapped = GptOssExpertWrapper(
                        expert_module=original_expert,
                        layer_idx=layer_idx,
                        expert_idx=expert_idx,
                        core_engine=self.core_engine,
                        swiglu_limit=swiglu_limit,
                        get_weights=False,
                        preloaded_weights=self.preloaded_expert_weights[module_key],
                    )
                    local_count += 1
                else:
                    # Remote expert: load dynamically via HtoD
                    wrapped = GptOssExpertWrapper(
                        expert_module=original_expert,
                        layer_idx=layer_idx,
                        expert_idx=expert_idx,
                        core_engine=self.core_engine,
                        swiglu_limit=swiglu_limit,
                        get_weights=True,
                        preloaded_weights=None,
                    )
                    dynamic_count += 1

                wrapped_experts.append(wrapped)

            # Replace original experts with wrapped versions
            mlp_block.experts = wrapped_experts

            # Store reference to core_engine on MLPBlock for potential direct use
            mlp_block.core_engine = self.core_engine

        logging.info(
            f"Wrapped {local_count + dynamic_count} expert modules: "
            f"{local_count} local (pre-loaded), {dynamic_count} dynamic (HtoD)"
        )

    def configure_decoding(self) -> Tuple:
        """Configure model for decoding phase.

        For GPT-OSS with single-GPU deployment:
        - Model is already initialized from prefill
        - Just update the phase to "decode" for all wrappers
        - Return the same model and weight_copy_task

        For multi-GPU deployment:
        - Similar to prefill but with phase="decode"
        - Expert routing may differ (more experts in host memory for decode)

        Returns:
            Tuple of (model, weight_copy_task)
        """
        # If model doesn't exist yet (shouldn't happen, but handle it)
        if not hasattr(self, 'model') or self.model is None:
            logging.warning("configure_decoding called before configure_prefill, initializing...")
            return self.configure_prefill()

        # Update phase for all wrappers
        self._set_decode_phase()

        logging.info("Configured for decoding phase")
        return self.model, self.weight_copy_task

    def _set_decode_phase(self):
        """Set phase to 'decode' for all attention and expert wrappers.

        This affects:
        - GptOssAttnWrapper.phase (class variable) - for weight prefetching
        - GptOssExpertWrapper.phase (instance variable) - for expert weight loading
        """
        # Set attention wrapper phase (class variable)
        GptOssAttnWrapper.set_phase("decode")

        # Set expert wrapper phase (instance variables)
        transformer = self.model.transformer if hasattr(self.model, 'transformer') else self.model
        num_layers = self.model_config.num_hidden_layers
        num_experts = self.model_config.num_local_experts

        for layer_idx in range(num_layers):
            mlp_block = transformer.block[layer_idx].mlp
            for expert_idx in range(num_experts):
                expert_wrapper = mlp_block.experts[expert_idx]
                if hasattr(expert_wrapper, 'set_phase'):
                    expert_wrapper.set_phase("decode")

    def set_prefill_phase(self):
        """Reset phase to 'prefill' for all wrappers.

        Called when switching back from decode to prefill mode.
        """
        # Set attention wrapper phase (class variable)
        GptOssAttnWrapper.set_phase("prefill")

        # Set expert wrapper phase (instance variables)
        transformer = self.model.transformer if hasattr(self.model, 'transformer') else self.model
        num_layers = self.model_config.num_hidden_layers
        num_experts = self.model_config.num_local_experts

        for layer_idx in range(num_layers):
            mlp_block = transformer.block[layer_idx].mlp
            for expert_idx in range(num_experts):
                expert_wrapper = mlp_block.experts[expert_idx]
                if hasattr(expert_wrapper, 'set_phase'):
                    expert_wrapper.set_phase("prefill")

    def get_weight_copy_task(self) -> Dict[str, List[str]]:
        """Return the weight copy task mapping."""
        return self.weight_copy_task

    def get_state_dict_name_map(self) -> Dict[str, Dict[str, str]]:
        """Return the state dict name mapping."""
        return self.state_dict_name_map
