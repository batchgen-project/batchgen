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

"""GPT-OSS-120B specific planner for BatchGen configuration.

Optimized for single H20 GPU deployment with MXFP4 quantized weights.
"""

from batchgen.planner.base_planner import BasePlanner


class GptOssPlanner(BasePlanner):
    """Planner specialized for GPT-OSS-120B model.

    Optimized for single H20 GPU (96GB) with MXFP4 quantization.

    Model characteristics:
    - 117B parameters (5.1B active)
    - 36 layers, hidden_size=2880
    - GQA: 64 heads, 8 KV heads
    - 128 experts, Top-4 routing
    - MXFP4 quantization (~55GB weights)
    """

    # Model-specific constants
    MAGIC_NUM = 90_000  # Optimized for GPT-OSS MoE architecture
    DEFAULT_MEM_FRAC = 0.90  # Higher utilization for single GPU
    NUM_EXPERTS = 128  # Total experts in GPT-OSS-120B
    NUM_LAYERS = 36

    def __version__(self):
        """Version for H20 single GPU and GPT-OSS-120B."""
        return "0.1.0"

    def _adjust_config_for_model(self):
        """GPT-OSS-specific config adjustments.

        For single H20 GPU (world_size == 1):
        - All 128 experts local (no expert parallelism needed)
        - Expert indices computed dynamically in model based on rank/world_size
        - MXFP4 experts are smaller than FP8 (~0.4GB vs 2.4GB each)

        For world_size == 1:
        - experts_per_rank = NUM_EXPERTS // 1 = 128
        - routed_expert_start_idx = 0 * 128 = 0
        - routed_expert_end_idx = 1 * 128 = 128

        GPT-OSS specific settings:
        - No shared experts (set buffer counts to 0)
        - Use attn_mode=3 for proper dynamic weight loading
        """
        # Override base planner's num_local_expert calculation
        # Base planner assumes 2.4GB per expert (FP8), but MXFP4 experts are ~0.4GB
        # For world_size == 1, all experts should be local
        if self.world_size == 1:
            # All 128 experts fit on single H20 with MXFP4 (~55GB total)
            self.config.EP_Config.num_local_expert_per_layer = self.NUM_EXPERTS
            # No module buffer swapping needed - all experts resident
            self.config.GPU_Buffer_Config.num_decoding_module_buffer["routed_expert"] = 0

        # GPT-OSS does NOT have shared experts - set buffer counts to 0
        self.config.GPU_Buffer_Config.num_prefill_module_buffer["shared_expert"] = 0
        self.config.GPU_Buffer_Config.num_decoding_module_buffer["shared_expert"] = 0

        # GPT-OSS uses attn_mode=3 for proper dynamic weight loading
        # (attn_mode=1 is legacy and doesn't support dynamic loading correctly)
        self.config.Basic_Config.attn_mode = 3

    def get_module_shapes(self) -> dict:
        """Return GPT-OSS-120B specific tensor shapes."""
        return {
            "hidden_size": 2880,
            "num_attention_heads": 64,
            "num_kv_heads": 8,  # GQA 8:1 ratio
            "head_dim": 64,
            "num_layers": 36,
            "num_experts": 128,
            "num_experts_per_tok": 4,  # Top-4 experts selected
            "expert_intermediate_size": 2880,  # SwiGLU intermediate
            "sliding_window": 128,  # For alternating attention
            "vocab_size": 201088,
            # MXFP4 specific
            "mxfp4_block_size": 32,  # FP4 values per scale
        }

    def get_memory_estimate(self) -> dict:
        """Estimate memory usage for GPT-OSS-120B on H20.

        Returns dict with memory estimates in GB.
        """
        return {
            "moe_weights_mxfp4": 55.0,  # 128 experts, MXFP4 quantized
            "attention_weights_bf16": 3.0,  # GQA attention (not quantized)
            "embeddings_lm_head": 2.0,  # Embeddings + LM head (not quantized)
            "kv_cache_32k_batch16": 3.0,  # KV cache estimate
            "activations_workspace": 8.0,  # Runtime buffers
            "total_estimate": 71.0,
            "h20_total": 96.0,
            "headroom": 25.0,
        }


# Alias for backward compatibility
Scheduler = GptOssPlanner
