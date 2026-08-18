"""Pure tensor slicing rules for Kimi-Linear attention-group TP weights."""


def _slice(tensor, name, dim, tp_size, tp_rank):
    if tp_size == 1:
        return tensor
    if tp_size <= 0 or not 0 <= tp_rank < tp_size:
        raise ValueError(f"invalid TP rank {tp_rank}/{tp_size} for {name}")
    width = tensor.shape[dim]
    if width % tp_size != 0:
        raise ValueError(
            f"tensor {name} shape {tuple(tensor.shape)} cannot be split "
            f"across TP={tp_size} on dim {dim}"
        )
    chunk = width // tp_size
    return tensor.narrow(dim, tp_rank * chunk, chunk).contiguous()


def shard_shared_expert_tensor(tensor, name, tp_size, tp_rank):
    """Return this TP rank's standard MLP weight shard.

    gate/up are column-parallel (output rows); down is row-parallel (input
    columns). Summing the per-rank down-projection outputs reconstructs the
    original shared-expert output.
    """
    base = name.split(".", 1)[0]
    if base in ("gate_proj", "up_proj"):
        dim = 0
    elif base == "down_proj":
        dim = 1
    else:
        raise ValueError(f"unsupported shared-expert tensor for TP: {name}")
    return _slice(tensor, name, dim, tp_size, tp_rank)


def shard_mla_tensor(tensor, name, tp_size, tp_rank):
    """Return this TP rank's NoPE-MLA weight shard.

    The low-rank q/kv-A projections and their norms are replicated. Projections
    that produce per-head values are column-parallel (output rows), while
    ``o_proj`` is row-parallel over the local value-head columns.
    """
    if tp_size == 1:
        return tensor

    base = name.split(".", 1)[0]
    if base in (
        "q_a_proj",
        "q_a_layernorm",
        "kv_a_proj_with_mqa",
        "kv_a_layernorm",
    ):
        if tp_size <= 0 or not 0 <= tp_rank < tp_size:
            raise ValueError(f"invalid MLA TP rank {tp_rank}/{tp_size}")
        return tensor
    if base in ("q_b_proj", "q_proj", "kv_b_proj", "g_proj"):
        dim = 0
    elif base == "o_proj":
        dim = 1
    else:
        raise ValueError(f"unsupported MLA tensor for TP: {name}")
    return _slice(tensor, name, dim, tp_size, tp_rank)
