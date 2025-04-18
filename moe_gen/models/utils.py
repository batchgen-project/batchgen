import torch
import torch.nn.functional as F
from einops import rearrange
from sgl_kernel import flash_ops

def is_fa3_supported(device=None) -> bool:
    #  now sgl-kernel only build fa3 for sm90a && cuda >= 12.3
    return (torch.cuda.get_device_capability(device)[0] == 9) and (
        torch.version.cuda >= "12.3"
    )

def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    seqused_q=None,
    seqused_k=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    sm_margin=0,
    return_softmax_lse=False,
):
    if not is_fa3_supported():
        raise NotImplementedError(
            "flash_attn at sgl-kernel is only supported on sm90 and above"
        )

    if softmax_scale is None:
        softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (
            -0.5
        )

    out, softmax_lse, *rest = torch.ops.sgl_kernel.fwd.default(
        q,
        k,
        v,
        None,  # k_new
        None,  # v_new
        qv,  # qv
        None,  # out
        cu_seqlens_q,
        cu_seqlens_k,
        None,  # cu_seqlens_k_new
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        None,  # page_table,
        None,  # kv_batch_idx
        None,  # leftpad_k
        None,  # rotary cos
        None,  # rotary sin
        None,  # seqlens_rotary
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        is_rotary_interleaved=False,
        scheduler_metadata=None,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
    )

    return (out, softmax_lse, *rest) if return_softmax_lse else out

def index_first_axis(input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    assert input.ndim >= 2, "Expected input with at least 2 dimensions"
    B = input.shape[0]
    other_shape = input.shape[1:]
    D = int(torch.prod(torch.tensor(other_shape)))  # 展平成向量维度

    input_flat = input.reshape(B, D)  # shape: [B, D]
    gather_idx = indices[:, None].expand(-1, D)  # shape: [N, D]
    out = torch.gather(input_flat, dim=0, index=gather_idx)  # shape: [N, D]
    return out.view(len(indices), *other_shape)  # shape: [N, ...]

def index_put_first_axis(values: torch.Tensor, indices: torch.Tensor, first_axis_dim: int) -> torch.Tensor:
    assert indices.ndim == 1, "indices must be 1D"
    assert values.shape[0] == indices.shape[0], "values and indices must have the same length"

    output = torch.zeros(
        (first_axis_dim, *values.shape[1:]),
        dtype=values.dtype,
        device=values.device
    )

    output[indices] = values

    return output


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    dim = hidden_states.shape[-1]
    # output = torch.zeros((batch * seqlen), dim, device=hidden_states.device, dtype=hidden_states.dtype)
    # output[indices] = hidden_states
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)

def get_valid_token_mask_from_causal_mask(attention_mask):
    """
    Get valid token mask from attention mask.
    Args:
        attention_mask (torch.Tensor): Attention mask of shape (B, 1, L, L).
    Returns:
        torch.Tensor: Valid token mask of shape (B, L).
    """
    B, _, L, _ = attention_mask.shape
    diag_indices = torch.arange(L, device=attention_mask.device)
    diag_values = attention_mask[:, 0, diag_indices, diag_indices]  # shape: [B, L]

    zero = torch.tensor(0.0, dtype=diag_values.dtype, device=diag_values.device)
    valid_mask = torch.isclose(diag_values, zero)  # True = valid token
    return valid_mask.to(torch.int32)              # shape: [B, L]