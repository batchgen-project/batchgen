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
    # lighter per-token compute than K2.5; keep H20 MoE magic num.
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

        # BF16 KV (no kv quantization). Canonical spelling is "bfloat16":
        # base_planner's kv_element_size check and the engine dtype map
        # (core/utils.cpp) both match on "bfloat16", never "bf16".
        self.config.Basic_Config.kv_dtype = "bfloat16"

        # F6: _compute_batch_configs() ran before this method with the base
        # default attn_mode=1 (our attn_mode=3 above lands after buffer
        # sizing), so its expert-residency math is wrong for this model.
        # Override all three results here — this method runs last, so the
        # values stick:
        # (a) Decode expert staging buffers: normalize to the prefill ring (8).
        #     The memory-based split left 7-9 buffers. Kept non-zero under
        #     resident-EP decode (which streams nothing — the worker only
        #     starts the decode H2D streamer when the PSM returns a non-empty
        #     routed_expert task): 8 slots ~ 113 MB keep the
        #     decode_moe_mode="streamed" fallback launchable without
        #     re-planning.
        self.config.GPU_Buffer_Config.num_decoding_module_buffer["routed_expert"] = 8
        # (b) Zero persistent experts FOR THE ENGINE: this field is consumed
        #     Python-side only (base_planner sizing, other models' PSMs; the
        #     C++ engine reads model_config.num_local_experts in
        #     core/utils.cpp:357, never EP_Config) — it does NOT size the
        #     resident-EP decode shards, which the Kimi PSM keeps fully
        #     Python-side. 0 states the engine-facing contract: all streamed
        #     (prefill streams all 256 experts/rank, persistent=False).
        self.config.EP_Config.num_local_expert_per_layer = 0
        # (c) Decode admission cap — consumed by the worker as the per-rank
        #     max decode sequences (batchgen_worker rank_counts checks); the
        #     base leaves it None -> TypeError on first decode admission.
        #     16/rank x 8 ranks = single-wave admission of a 128-seq batch.
        #     Also bounds the resident-EP decode collective layout
        #     (num_tokens_per_rank <= 16 -> <= 128-row global buffer).
        self.config.Module_Batching_Config.MoE_decoding_micro_batch_size = 16

        # M4 P0.3 — decode MoE execution mode, read by the PSM at
        # configure_decoding (config-driven, no env vars):
        #   "resident_ep": stacked EP-8 BF16 shards resident on GPU
        #                  (~11.8 GB/rank) + fused_moe_bf16 + all_reduce;
        #   "streamed":    legacy per-expert host streaming (fallback;
        #                  ~94 GB H2D/step -> ~0.5 tok/s floor).
        self.config.EP_Config.decode_moe_mode = "resident_ep"

        # KDA conv/recurrent state-pool slots (peak concurrent sequences the
        # PSM pools can hold). Config-driven — replaces the retired
        # BATCHGEN_KDA_STATE_SLOTS env var (M1-C).
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
