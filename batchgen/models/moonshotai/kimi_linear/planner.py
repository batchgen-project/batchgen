# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#  see the license at                                                           #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
# ---------------------------------------------------------------------------- #

"""Kimi-Linear specific planner for BatchGen configuration."""

from batchgen.planner.base_planner import BasePlanner


class KimiLinearPlanner(BasePlanner):
    """Planner specialized for Kimi-Linear (48B-A3B, KDA + NoPE-MLA hybrid).

    - 27 layers: 20 KDA (linear attention, no KV) + 7 NoPE-MLA (paged KV,
      kv_lora_rank 512 + qk_rope_head_dim 64 per token, BF16).
    - 256 BF16 routed experts (moe_intermediate 1024) + 1 shared, top-8.
    - KV is only stored for MLA layers; KDA state lives in a separate
      Python-side pool (KDAStateGPUManager), not in KV_Storage.
    """

    # 256 experts * 3 projections * 1024 moe_intermediate / 8 top-k —
    # lighter per-token compute than K2.5; keep K2.5's MoE magic num.
    MAGIC_NUM = 672_000
    DEFAULT_MEM_FRAC = 0.85
    NUM_EXPERTS = 256

    def __version__(self):
        return "0.1.0"

    def _adjust_config_for_model(self):
        """Kimi-Linear specific config adjustments."""
        # Modern paged-KV decode path (decoding_continuous), same as K2.5.
        self.config.Basic_Config.attn_mode = 3

        # Kimi-Linear context: model_max_length=1048576; keep the micro-batch
        # token cap aligned with the KV/prefill budget instead of the raw
        # context window.
        self.config.Module_Batching_Config.prefill_micro_batch_token_cap = 262_144

        # BF16 KV (no kv quantization).
        self.config.Basic_Config.kv_dtype = "bf16"

        # KDA conv/recurrent state-pool slots (peak concurrent sequences the
        # PSM pools can hold). Config-driven (M1-C).
        self.config.GPU_Buffer_Config.kda_state_slots = 256

    def get_module_shapes(self) -> dict:
        """Return Kimi-Linear specific tensor shapes."""
        return {
            "hidden_size": 2304,
            "num_attention_heads": 32,
            "num_kv_heads": 1,  # MLA — all heads compress KV
            "num_layers": 27,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "expert_hidden_size": 1024,  # moe_intermediate_size
            "kv_lora_rank": 512,
            "q_lora_rank": None,  # direct q_proj (no q lora)
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
        }
