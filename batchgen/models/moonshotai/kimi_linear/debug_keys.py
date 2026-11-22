"""Server-level debug keys read by Kimi code (see ``batchgen/debug_state.py``)."""
from batchgen.debug_state import TIER_REWARM, key, parse_bool, parse_choice

DEBUG_KEYS = (
    key("k3_triton_strict", "kimi_linear", "both", parse_bool,
         "raise on a Kimi kernel compile after readiness"),
    key("k3_seam_verify_bytes", "kimi_linear", "prefill", parse_bool,
         "byte-check streamed weight transfers at the seam"),
    key("k3_prefill_routed_kernel", "kimi_linear", "prefill", parse_choice("dequant_once", "marlin"),
         "routed-expert kernel of streamed prefill"),
    key("k3_prefill_finite_check", "kimi_linear", "prefill", parse_bool,
         "check prefill activations for non-finite values"),
    key("k3_prefill_profile", "kimi_linear", "prefill", parse_bool,
         "per-layer prefill profile; part of the prefill instance key",
         tier=TIER_REWARM),
    key("kimi_decode_graph_mode", "kimi_linear", "decode", parse_choice("graph", "eager", "compare"),
         "run decode from captured graphs, eagerly, or both and compare",
         tier=TIER_REWARM),
)

RETIRED_DEBUG_KEYS = {
    "k3_prefill_moe_mode": "the prefill MoE mode is the launch parameter --prefill-moe-mode",
    "k3_decode_profile_steps": "renamed decode_profile_steps",
    "k3_decode_profile_ranks": "renamed decode_profile_ranks",
    "k3_decode_profile_trace": "renamed decode_profile_trace",
}
