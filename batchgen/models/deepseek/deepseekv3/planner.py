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
        """DeepSeek-R1/V3 config adjustments.

        The base-class memory heuristic (available_memory_for_expert_cache // 2.4) over-counts
        FP8 expert size and, on single-node without offloading (attn_mode=1), caps persistent
        experts below expert_per_rank. With offloading off, the remaining routed experts are
        never loaded, so decode is silently wrong (see planner bug: num_local=26 vs 32/rank).

        DeepSeek-R1 FP8 (256 experts / 8 ranks = 32/rank) fits resident on H20. Mirror the Kimi
        planner: force all experts persistent and use the modern paged decode path (attn_mode=3).
        This also matches the dual-node config (16/rank), which the base class already produces.

        Only applies when EP offloading is OFF; when offloading is enabled the base-class
        persistent/host split is respected.
        """
        if self.config.EP_Config.enable_offloading:
            return

        expert_per_rank = self.NUM_EXPERTS // self.world_size
        self.config.EP_Config.num_local_expert_per_layer = expert_per_rank
        # Modern paged decode path (decoding_attn_mode_3_bf16); BF16 paged KV via the
        # gpu_paged_kv_manager. All experts persistent => no routed-expert decode buffers.
        self.config.Basic_Config.attn_mode = 3
        self.config.GPU_Buffer_Config.num_decoding_module_buffer["routed_expert"] = 0

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
