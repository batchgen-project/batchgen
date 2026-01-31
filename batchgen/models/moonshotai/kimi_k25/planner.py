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

"""Kimi K2.5 specific planner for BatchGen configuration."""

from batchgen.planner.base_planner import BasePlanner


class KimiK25Planner(BasePlanner):
    """Planner specialized for Kimi K2.5 models.

    Kimi K2.5 is a DeepSeek-V3 variant with 384 routed experts.
    Same MLA attention, same H20 target.
    """

    # Model-specific constants
    # 384 experts * 3 projections * 2048 moe_intermediate / 8 top-k ≈ similar compute density
    # Use similar magic num as DeepSeek-V3 scaled for 384 experts
    MAGIC_NUM = 672_000  # Same as DeepSeek-V3 (both MoE on H20)
    DEFAULT_MEM_FRAC = 0.85
    NUM_EXPERTS = 384  # K2.5 has 384 routed experts (vs 256 for DeepSeek-V3)

    def __version__(self):
        """Version for H20 and Kimi K2.5."""
        return "0.1.0"

    def _adjust_config_for_model(self):
        """K2.5-specific config adjustments.

        Most config is set by base class. This handles K2.5 overrides if needed.
        """
        pass

    def get_module_shapes(self) -> dict:
        """Return Kimi K2.5 specific tensor shapes."""
        return {
            "hidden_size": 7168,
            "num_attention_heads": 64,
            "num_kv_heads": 64,  # MLA — all heads compress KV
            "num_layers": 61,
            "num_experts": 384,
            "num_experts_per_tok": 8,
            "expert_hidden_size": 2048,  # moe_intermediate_size
            "kv_lora_rank": 512,
            "q_lora_rank": 1536,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
        }
