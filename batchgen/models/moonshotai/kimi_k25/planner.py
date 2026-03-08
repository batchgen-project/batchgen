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

        K2.5 uses INT4 W4A16 experts which are ~30MB each (not 2.4GB like FP8/BF16).
        Override expert memory calculations.
        """
        # INT4 expert size: ~30MB per expert (vs 2.4GB for FP8/BF16)
        # With 384 experts / 8 ranks = 48 experts per rank
        # 48 * 0.03GB = 1.44GB total expert cache (fits easily in GPU memory)
        expert_per_rank = self.NUM_EXPERTS // self.world_size  # 48

        # For K2.5 INT4, all experts can fit in GPU memory
        # Override the memory-based calculation from base class
        self.config.EP_Config.num_local_expert_per_layer = expert_per_rank

        # No need for expert offloading buffers - all experts are persistent
        self.config.GPU_Buffer_Config.num_decoding_module_buffer["routed_expert"] = 0

        # K2.5 requires attn_mode=3 for modern decoding path (decoding_continuous)
        # Override base planner logic that sets attn_mode=1 when enable_offloading=False
        self.config.Basic_Config.attn_mode = 3

        # EP offloading is disabled for K2.5 (all experts resident on GPU)
        self.config.EP_Config.enable_offloading = False

        # K2.5 context window: max_position_embeddings=262144 (YaRN: 4096 * factor=64)
        self.config.Module_Batching_Config.prefill_micro_batch_token_cap = 262_144

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
