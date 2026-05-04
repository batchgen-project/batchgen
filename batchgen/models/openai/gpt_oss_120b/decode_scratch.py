"""GPT-OSS decode scratch-memory reservation estimates."""

from __future__ import annotations

from typing import Any


def estimate_gpt_oss_decode_scratch_reserve_gb(
    *,
    model_config: Any,
    world_size: int,
    max_num_seq_per_rank: int,
) -> float:
    """Estimate non-KV HBM reserve needed by GPT-OSS decode kernels."""
    model_type = getattr(model_config, "model_type", "")
    if "gpt_oss" not in model_type:
        return 0.0

    max_num_seq_per_rank = max(int(max_num_seq_per_rank), 1)
    global_tokens = max_num_seq_per_rank * max(int(world_size), 1)
    hidden_size = int(getattr(model_config, "hidden_size", 2880))
    intermediate_size = int(
        getattr(model_config, "intermediate_size", hidden_size)
    )
    num_experts_per_tok = int(getattr(model_config, "num_experts_per_tok", 4))
    num_local_experts = int(getattr(model_config, "num_local_experts", 128))
    vocab_size = int(getattr(model_config, "vocab_size", 201088))

    bytes_per_bf16 = 2
    bytes_per_fp32 = 4
    moe_activation_bytes = (
        3
        * global_tokens
        * num_experts_per_tok
        * max(hidden_size, intermediate_size)
        * bytes_per_bf16
    )
    router_bytes = (
        global_tokens
        * num_local_experts
        * (bytes_per_bf16 + bytes_per_fp32)
    )
    topk_bytes = (
        global_tokens
        * num_experts_per_tok
        * (bytes_per_fp32 + bytes_per_fp32)
    )
    logits_bytes = max_num_seq_per_rank * vocab_size * bytes_per_bf16
    sampling_bytes = min(max_num_seq_per_rank, 64) * vocab_size * bytes_per_fp32

    estimated_gb = (
        moe_activation_bytes
        + router_bytes
        + topk_bytes
        + logits_bytes
        + sampling_bytes
    ) / (1024**3)

    return max(2.0, estimated_gb * 1.5)
