"""Byte-accounting formulas for the legacy GPU microbatch decoder."""


def legacy_past_kv_byte_size(
    model_config,
    *,
    max_input_length,
    token_idx,
    max_cache_len,
) -> int:
    """Return the legacy decoder's estimated past-KV byte size."""
    model_type = model_config.model_type
    if "deepseek" in model_type:
        return (
            (max(max_input_length + token_idx + 1, max_cache_len))
            * model_config.compressed_kv_dim
        )
    if "mixtral" in model_type:
        return (
            (max(max_input_length + token_idx, max_cache_len))
            * model_config.num_key_value_heads
            * model_config.head_dim
            * 2
        )
    raise ValueError(f"Model architecture {model_type} not supported yet.")
