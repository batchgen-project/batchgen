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

"""DeepSeek-V3/R1 specific planner for BatchGen configuration."""

from batchgen.planner.base_planner import BasePlanner


class DeepSeekV3Planner(BasePlanner):
    """Planner specialized for DeepSeek-V3/R1 models.

    Optimized for H20 GPUs with MoE architecture.
    """

    # Model-specific constants
    MAGIC_NUM = 672_000  # 224000 * 3, optimized for DeepSeek MoE architecture
    DEFAULT_MEM_FRAC = 0.85  # Higher utilization for DeepSeek
    NUM_EXPERTS = 256  # Total experts in DeepSeek-V3/R1

    def __version__(self):
        """Version for H20 and DeepSeek-R1/V3."""
        return "0.1.6"

    def _adjust_config_for_model(self):
        """DeepSeek-specific config adjustments.

        Most config is already set by base class _compute_batch_configs().
        This method handles any DeepSeek-specific overrides if needed.
        """
        # Currently no additional adjustments needed beyond base class
        # The base class handles:
        # - EP config (num_local_expert_per_layer)
        # - Attention mode (1 for single-node, 3 for dual-node)
        # - Decoding buffers
        # - Prefill batch sizes based on prompt length
        pass

    def get_module_shapes(self) -> dict:
        """Return DeepSeek-V3/R1 specific tensor shapes."""
        return {
            "hidden_size": 7168,
            "num_attention_heads": 128,
            "num_kv_heads": 8,  # MLA uses 8 KV heads
            "num_layers": 61,
            "num_experts": 256,
            "num_experts_per_tok": 8,  # Top-8 experts selected
            "expert_hidden_size": 2048,
            "kv_lora_rank": 512,  # MLA KV compression rank
            "q_lora_rank": 1536,  # MLA Q compression rank
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
        }


# Alias for backward compatibility
Scheduler = DeepSeekV3Planner
