"""Causal conv1d kernels — varlen prefill + pooled-state decode update.

Ported from sgl-kernel (csrc/mamba/causal_conv1d.cu) / Dao-AILab causal-conv1d.

Layouts and state semantics (kernel width W, conv-state width W-1):
  - Prefill pool layout: (num_slots, dim, W-1); a slot holds the last W-1
    inputs [x_{T-W+1} .. x_{T-1}] at positions [0 .. W-2].
  - fla ShortConvolution cache [N, D, W] maps as: fla_state[..., 1:] ==
    cuda_state[..., 0:W-1].
  - Slots equal to pad_slot_id are skipped (state untouched; prefill output
    rows for padded sequences are left as the staged input values).

Usage (BatchGen token-major serving layout):

    from batchgen_kernels.conv1d import causal_conv1d_fwd, causal_conv1d_update

    # prefill: x (total_tokens, dim) -> y (total_tokens, dim) [strided view]
    y = causal_conv1d_fwd(
        x, weight, bias=bias,
        conv_states=pool,            # (num_slots, dim, W-1), final state in place
        query_start_loc=cu_seqlens,  # (batch+1,) int32
        cache_indices=slot_ids,      # (batch,) int32
        has_initial_state=has_init,  # (batch,) bool or None
    )

    # decode: x (batch, dim) -> y (batch, dim); conv_state pool updated in place
    y = causal_conv1d_update(
        x, pool, weight, bias=bias, conv_state_indices=slot_ids,
    )
"""

import torch

from batchgen_kernels import load_extension

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        _ext = load_extension("batchgen_kernels.conv1d._C_causal_conv1d")
    return _ext


def _prep_weight(weight: torch.Tensor) -> torch.Tensor:
    # Accept (dim, W) or fla's (dim, 1, W); return contiguous (dim, W).
    if weight.dim() == 3:
        weight = weight.squeeze(1)
    return weight.contiguous()


def causal_conv1d_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    conv_states: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    silu_activation: bool = True,
    pad_slot_id: int = -1,
) -> torch.Tensor:
    """Causal conv1d prefill.

    Args:
        x: (total_tokens, dim) token-major packed varlen input (requires
            query_start_loc), or (batch, dim, seqlen) channel-major batched
            input. A staging copy is made; x is not modified.
        weight: (dim, W) or (dim, 1, W).
        bias: optional (dim,).
        conv_states: optional (num_slots, dim, W-1) pool; final state of each
            sequence is written in place at cache_indices.
        query_start_loc: (batch+1,) int32 cumulative sequence lengths.
        cache_indices: (batch,) int32 pool slot per sequence.
        has_initial_state: (batch,) bool; if set, the slot's current content is
            used as the conv initial state (chunked-prefill continuation).
        silu_activation: apply SiLU (default True).
        pad_slot_id: slot id treated as padding (skipped).

    Returns:
        Output in the same logical layout as x: (total_tokens, dim) view over
        the channel-major staging buffer for varlen input, or
        (batch, dim, seqlen) for batched input.
    """
    ext = _get_ext()
    weight = _prep_weight(weight)
    if bias is not None:
        bias = bias.contiguous()
    if query_start_loc is not None:
        # token-major (total, dim) -> channel-major (dim, total) staging copy
        assert x.dim() == 2, "varlen x must be (total_tokens, dim)"
        x_cm = x.t().contiguous()
        ext.causal_conv1d_fwd(
            x_cm, weight, bias, conv_states,
            query_start_loc.to(torch.int32) if query_start_loc.dtype != torch.int32 else query_start_loc,
            cache_indices.to(torch.int32) if cache_indices is not None and cache_indices.dtype != torch.int32 else cache_indices,
            has_initial_state,
            silu_activation, pad_slot_id,
        )
        return x_cm.t()  # (total, dim) strided view; downstream handles strides
    assert x.dim() == 3, "batched x must be (batch, dim, seqlen)"
    x_c = x if x.stride(-1) == 1 else x.contiguous()
    ext.causal_conv1d_fwd(
        x_c, weight, bias, conv_states, None,
        cache_indices.to(torch.int32) if cache_indices is not None and cache_indices.dtype != torch.int32 else cache_indices,
        has_initial_state,
        silu_activation, pad_slot_id,
    )
    return x_c


def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    silu_activation: bool = True,
    cache_seqlens: torch.Tensor | None = None,
    conv_state_indices: torch.Tensor | None = None,
    pad_slot_id: int = -1,
) -> torch.Tensor:
    """Causal conv1d decode update (functional; x is not modified).

    Args:
        x: (batch, dim) single-token or (batch, dim, seqlen) multi-token input.
        conv_state: (num_slots, dim, state_len) pool, state_len >= W-1;
            updated in place at conv_state_indices.
        weight: (dim, W) or (dim, 1, W).
        bias: optional (dim,).
        silu_activation: apply SiLU (default True).
        cache_seqlens: optional (batch,) int32; if set, conv_state is treated
            as a circular buffer indexed by these positions.
        conv_state_indices: (batch,) int32 pool slot per request.
        pad_slot_id: slot id treated as padding (skipped; output rows are 0).

    Returns:
        Output with the same shape as x ((batch, dim) or (batch, dim, seqlen)).
    """
    ext = _get_ext()
    weight = _prep_weight(weight)
    if bias is not None:
        bias = bias.contiguous()
    return ext.causal_conv1d_update(
        x, conv_state, weight, bias, silu_activation,
        cache_seqlens,
        conv_state_indices.to(torch.int32) if conv_state_indices is not None and conv_state_indices.dtype != torch.int32 else conv_state_indices,
        pad_slot_id,
    )
