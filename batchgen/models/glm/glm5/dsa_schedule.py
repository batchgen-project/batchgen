"""Pure helpers for the GLM-5.2 DSA indexer reuse schedule."""


def dsa_layer_skips_topk(config, layer_id: int) -> bool:
    """Whether ``layer_id`` reuses the previous full layer's DSA top-k indices.

    The frequency/offset formula matches SGLang.  GLM-5 and GLM-5.1 configs
    have no positive reuse frequency, so every layer recomputes top-k.
    """
    pattern = getattr(config, "index_topk_pattern", None)
    if pattern is not None:
        return layer_id < len(pattern) and pattern[layer_id] == "S"

    freq = getattr(config, "index_topk_freq", None)
    if freq is None:
        freq = 1
    if freq <= 0:
        raise ValueError(f"index_topk_freq must be positive, got {freq}")
    if freq == 1:
        return False

    offset = getattr(config, "index_skip_topk_offset", None)
    if offset is not None:
        if offset <= 0:
            raise ValueError(
                "index_skip_topk_offset must be positive; offset <= 0 marks "
                "layer 0 as skip_topk with no prior topk to reuse"
            )
        return max(layer_id - offset + 1, 0) % freq != 0

    return max(layer_id - 1, 0) % freq != 0
