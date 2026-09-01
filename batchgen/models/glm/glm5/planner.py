# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
# ---------------------------------------------------------------------------- #

"""GLM-5 specific planner for BatchGen configuration."""

import logging

import torch

from batchgen.planner.base_planner import BasePlanner


class GLM5Planner(BasePlanner):
    """Planner specialized for GLM-5 models.

    Key differences from DeepSeek-V3:
    - 78 layers (vs 61)
    - hidden_size=6144 (vs 7168)
    - q_lora_rank=2048 (vs 1536)
    - qk_nope_head_dim=192, v_head_dim=256
    - 75 MoE layers (vs 60), expert size 36MB (vs ~40MB)
    """

    MAGIC_NUM = 672_000
    DEFAULT_MEM_FRAC = 0.85
    NUM_EXPERTS = 256

    # GLM-5 specific constants
    NUM_LAYERS = 78
    COMPRESSED_KV_DIM = 576  # kv_lora_rank(512) + qk_rope_head_dim(64)

    def __init__(self, model_name: str = ""):
        super().__init__()
        self.is_fp8 = "fp8" in model_name.lower()
        if self.is_fp8:
            self.EXPERT_SIZE_GB = 2.7   # 75 MoE layers * 36MB FP8
        else:
            self.EXPERT_SIZE_GB = 5.4   # 75 MoE layers * 72MB BF16

    def __version__(self):
        return "0.1.0"

    def _adjust_config_for_model(self):
        """GLM-5 specific config adjustments."""
        if self.is_fp8:
            # Pure-DP grouped prefill acquires every one of the 256 routed
            # experts for a layer at once. Two layers let the core-engine H2D
            # worker fill L+1 while L computes. Shared experts use the same
            # two-layer event-retired pipeline.
            self.config.GPU_Buffer_Config.num_prefill_module_buffer[
                "routed_expert"
            ] = 2 * self.NUM_EXPERTS
            self.config.GPU_Buffer_Config.num_prefill_module_buffer[
                "shared_expert"
            ] = 2

    def _compute_batch_configs(self):
        """Compute batch sizes with GLM-5 specific constants.

        Overrides base class which hardcodes DeepSeek values:
        - 61 layers -> 78 layers for per-sequence KV size
        - 2.4 GB/expert -> 2.7 GB/expert for memory budgeting
        """
        kv_element_size = 2 if self.config.Basic_Config.kv_dtype == "bfloat16" else 1

        expert_per_rank = self.NUM_EXPERTS // self.world_size
        assert expert_per_rank > 0, "EXPERT_PER_RANK must be greater than 0"

        # GLM-5 decode must always use the continuous attention path; legacy
        # attn modes do not support the DSA architecture.
        self.config.Basic_Config.attn_mode = 3

        attn_decoding_micro_batch_size = self.MAGIC_NUM // self.max_prompt_length
        attn_decoding_micro_batch_size = 2 ** (attn_decoding_micro_batch_size.bit_length() - 1)

        num_k_buffer = 6
        k_buffer_size = (
            num_k_buffer * attn_decoding_micro_batch_size *
            self.max_context_length * self.COMPRESSED_KV_DIM / (1024 ** 3) * kv_element_size
        )
        self.config.GPU_Buffer_Config.kv_buffer_num_tokens = (
            attn_decoding_micro_batch_size * self.max_context_length
        )

        # Query the actual device instead of assuming the H20's 96 GB: on
        # 141 GB parts the hardcoded value made attn_mode=3 refuse single-node
        # world_size=8 (32 experts/rank) even though it fits with margin.
        gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        available_gpu_mem = gpu_total_gb * self.DEFAULT_MEM_FRAC
        model_skeleton_size = 6
        cuda_page_table_default_size = 5
        nccl_default_buffer_usage = 2.5

        non_static_memory_usage = k_buffer_size + model_skeleton_size + cuda_page_table_default_size
        available_memory_for_expert_cache = available_gpu_mem - non_static_memory_usage

        if self.config.EP_Config.enable_offloading and self.config.EP_Config.offloading_ratio > 0:
            num_local_expert_per_layer = int(expert_per_rank * (1 - self.config.EP_Config.offloading_ratio))
            num_offloaded = expert_per_rank - num_local_expert_per_layer
            num_decoding_module_buffer_routed_expert = num_offloaded + 2
            logging.info(
                f"EP offloading enabled: {num_local_expert_per_layer} persistent, "
                f"{num_offloaded} offloaded, {num_decoding_module_buffer_routed_expert} buffers"
            )
        else:
            num_local_expert_per_layer = min(
                expert_per_rank,
                int(available_memory_for_expert_cache // self.EXPERT_SIZE_GB)
            )
            num_decoding_module_buffer_routed_expert = expert_per_rank - num_local_expert_per_layer + 2

        if self.config.Basic_Config.attn_mode == 3:
            if not self.config.EP_Config.enable_offloading:
                num_local_expert_per_layer = expert_per_rank

            expert_size = num_local_expert_per_layer * self.EXPERT_SIZE_GB
            moe_decode_memory_budget = (
                available_gpu_mem
                - model_skeleton_size
                - cuda_page_table_default_size
                - expert_size
                - nccl_default_buffer_usage
            )
            if moe_decode_memory_budget <= 0:
                raise RuntimeError(
                    "GLM-5 attn_mode=3 requires all local MoE experts to be "
                    f"persistent, but world_size={self.world_size} requires "
                    f"{expert_per_rank} experts/rank and exceeds the planner "
                    f"memory budget by {-moe_decode_memory_budget:.2f} GB. "
                    "Use the validated two-node H20 configuration or an "
                    "explicit offloading plan."
                )
            per_seq_size = (
                self.max_context_length * self.NUM_LAYERS * self.COMPRESSED_KV_DIM
                / (1024 ** 3) * kv_element_size
            )
            self.config.Module_Batching_Config.MoE_decoding_micro_batch_size = int(
                moe_decode_memory_budget / per_seq_size
            )
            logging.info(
                f"Max Available MoE decoding micro batch size: "
                f"{self.config.Module_Batching_Config.MoE_decoding_micro_batch_size}"
            )

        if num_local_expert_per_layer == expert_per_rank and not self.config.EP_Config.enable_offloading:
            num_decoding_module_buffer_routed_expert = 0

        self.config.Module_Batching_Config.attn_decoding_micro_batch_size = attn_decoding_micro_batch_size
        self.config.GPU_Buffer_Config.num_k_buffer = num_k_buffer
        self.config.GPU_Buffer_Config.num_decoding_module_buffer["routed_expert"] = num_decoding_module_buffer_routed_expert
        self.config.EP_Config.num_local_expert_per_layer = num_local_expert_per_layer

        if self.config.Basic_Config.attn_mode == 3 and not self.config.EP_Config.enable_offloading:
            self.config.GPU_Buffer_Config.num_decoding_module_buffer["routed_expert"] = 0

    def get_module_shapes(self) -> dict:
        """Return GLM-5 specific tensor shapes."""
        return {
            "hidden_size": 6144,
            "num_attention_heads": 64,
            "num_kv_heads": 64,
            "num_layers": 78,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "expert_hidden_size": 2048,
            "kv_lora_rank": 512,
            "q_lora_rank": 2048,
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "v_head_dim": 256,
        }
