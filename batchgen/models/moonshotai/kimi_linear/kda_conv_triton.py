# ---------------------------------------------------------------------------- #
#  BatchGen                                                                     #
#  copyright (c) EfficientMoE team 2025                                         #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""Token-major causal depthwise conv1d for the Kimi-K3 KDA prefill.

The production ``causal_conv1d_fwd`` kernel is channel-major: for a packed
prefill it sweeps each (sequence, channel) with one CTA and transposes the
result back into the token-major buffer. At exact 64K (524,288 x 1,536 bf16
per rank) that costs 20--26 ms per conv against a 0.8 ms read+write floor,
three times per KDA layer. This kernel reads the token-major ``[T, C]``
activations directly: one program per (token block, channel block), a
``width-1`` token halo, and the same fp32 tap order, bias,
``x / (1 + exp(-x))`` SiLU and bf16 rounding as the CUDA kernel, so the
results are bit-identical (pinned by
``tests/gpu/test_kimi_k3_kda_conv_triton.py``).

Semantics mirrored from ``batchgen_kernels/src/conv1d/causal_conv1d.cu``:

* sequence ``s`` covers rows ``[cu[s], cu[s+1])``; taps before ``cu[s]`` read
  ``conv_states[slot]`` when ``has_initial_state[s]`` else zeros;
* after the sweep ``conv_states[slot, :, k]`` holds raw input row
  ``end - 3 + k`` of the sequence;
* a sequence whose slot is ``pad_slot_id`` is left untouched.

Imported lazily by :mod:`serving_modules` for supported CUDA inputs only.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


_WIDTH = 4
_BLOCK_T = 64
_BLOCK_C = 64


@triton.jit
def _kda_conv_kernel(
    x_ptr,
    out_ptr,
    w_ptr,
    b_ptr,
    cu_ptr,
    slot_ptr,
    init_ptr,
    state_ptr,
    seq_of_block_ptr,
    base_of_block_ptr,
    stride_x_t,
    stride_out_t,
    stride_state_slot,
    stride_state_c,
    stride_state_l,
    C,
    pad_slot_id,
    HAS_INIT: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """One (token block, channel block) tile of the causal width-4 conv."""
    tb = tl.program_id(0)
    cb = tl.program_id(1)
    seq = tl.load(seq_of_block_ptr + tb)
    base = tl.load(base_of_block_ptr + tb)
    start = tl.load(cu_ptr + seq)
    end = tl.load(cu_ptr + seq + 1)
    slot = tl.load(slot_ptr + seq)
    rows = base + tl.arange(0, BLOCK_T)
    cols = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    row_ok = rows < end
    col_ok = cols < C
    mask = row_ok[:, None] & col_ok[None, :]
    if slot == pad_slot_id:
        # Padding sequence: the CUDA kernel skips it, so the buffer keeps the
        # raw input rows.
        raw = tl.load(
            x_ptr + rows[:, None] * stride_x_t + cols[None, :],
            mask=mask, other=0.0,
        )
        tl.store(out_ptr + rows[:, None] * stride_out_t + cols[None, :], raw, mask=mask)
        return

    if HAS_INIT:
        has_init = tl.load(init_ptr + seq)
    else:
        has_init = 0

    w0 = tl.load(w_ptr + cols * 4 + 0, mask=col_ok, other=0.0).to(tl.float32)
    w1 = tl.load(w_ptr + cols * 4 + 1, mask=col_ok, other=0.0).to(tl.float32)
    w2 = tl.load(w_ptr + cols * 4 + 2, mask=col_ok, other=0.0).to(tl.float32)
    w3 = tl.load(w_ptr + cols * 4 + 3, mask=col_ok, other=0.0).to(tl.float32)
    bias = tl.load(b_ptr + cols, mask=col_ok, other=0.0).to(tl.float32)

    # tap k reads row (t - (3 - k)); rows before ``start`` come from the
    # initial state (row start-3+j -> state index j) or are zero.
    acc = tl.zeros([BLOCK_T, BLOCK_C], tl.float32) + bias[None, :]
    for k in tl.static_range(0, 4):
        src = rows - (3 - k)
        in_seq = (src >= start) & row_ok
        xv = tl.load(
            x_ptr + src[:, None] * stride_x_t + cols[None, :],
            mask=in_seq[:, None] & col_ok[None, :],
            other=0.0,
        ).to(tl.float32)
        if HAS_INIT:
            before = (src < start) & row_ok
            sidx = src - (start - 3)
            sv = tl.load(
                state_ptr
                + slot * stride_state_slot
                + cols[None, :] * stride_state_c
                + sidx[:, None] * stride_state_l,
                mask=before[:, None] & col_ok[None, :] & (has_init != 0),
                other=0.0,
            ).to(tl.float32)
            xv = tl.where(before[:, None], sv, xv)
        if k == 0:
            wk = w0
        elif k == 1:
            wk = w1
        elif k == 2:
            wk = w2
        else:
            wk = w3
        acc = tl.math.fma(wk[None, :], xv, acc)

    y = tl.math.div_rn(acc, 1.0 + libdevice.exp(-acc))
    tl.store(
        out_ptr + rows[:, None] * stride_out_t + cols[None, :],
        y.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit
def _kda_conv_state_kernel(
    x_ptr,
    cu_ptr,
    slot_ptr,
    init_ptr,
    state_ptr,
    stride_x_t,
    stride_state_slot,
    stride_state_c,
    stride_state_l,
    C,
    pad_slot_id,
    HAS_INIT: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Write the last three raw input rows of each sequence to its slot.

    A sequence shorter than three tokens keeps the tail of its initial state
    (the CUDA kernel's halo), so index ``k`` of the new state is old index
    ``k + len`` when that position precedes the sequence.
    """
    seq = tl.program_id(0)
    cb = tl.program_id(1)
    start = tl.load(cu_ptr + seq)
    end = tl.load(cu_ptr + seq + 1)
    slot = tl.load(slot_ptr + seq)
    if slot == pad_slot_id:
        return
    if HAS_INIT:
        has_init = tl.load(init_ptr + seq)
    else:
        has_init = 0
    cols = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    col_ok = cols < C
    state_base = state_ptr + slot * stride_state_slot + cols * stride_state_c
    for k in tl.static_range(0, 3):
        src = end - 3 + k
        in_seq = src >= start
        v = tl.load(x_ptr + src * stride_x_t + cols, mask=col_ok & in_seq, other=0.0)
        if HAS_INIT:
            # old index for a pre-sequence position: src - (start - 3) > k,
            # so it is read before this loop overwrites it.
            sidx = src - (start - 3)
            sv = tl.load(
                state_base + sidx * stride_state_l,
                mask=col_ok & (in_seq == 0) & (has_init != 0),
                other=0.0,
            )
            v = tl.where(in_seq, v, sv)
        tl.store(state_base + k * stride_state_l, v, mask=col_ok)


def _weight_2d(weight):
    """``(dim, W)`` view of a ``(dim, W)`` or depthwise ``(dim, 1, W)`` weight."""
    if weight.ndim == 3 and weight.shape[1] == 1:
        return weight.reshape(weight.shape[0], weight.shape[2])
    return weight


def supports_kda_conv_triton(x, weight, conv_states) -> bool:
    """Return whether the production K3 conv shape can use the Triton path."""
    weight = _weight_2d(weight)
    return (
        x.is_cuda
        and x.ndim == 2
        and x.dtype == torch.bfloat16
        and x.stride(1) == 1
        and weight.ndim == 2
        and weight.shape[1] == _WIDTH
        and weight.shape[0] == x.shape[1]
        and conv_states is not None
        and conv_states.ndim == 3
        and conv_states.shape[1] == x.shape[1]
        and conv_states.shape[2] == _WIDTH - 1
        and conv_states.dtype == x.dtype
    )


def _block_plan(cu_list, block_t):
    """Per token block: owning sequence and the block's first row."""
    seq_of_block = []
    base_of_block = []
    for s in range(len(cu_list) - 1):
        start, end = cu_list[s], cu_list[s + 1]
        for b in range(start, end, block_t):
            seq_of_block.append(s)
            base_of_block.append(b)
    return seq_of_block, base_of_block


# One-entry cache keyed on the microbatch's cu_seqlens object (the worker
# builds it once per microbatch and every layer receives the same tensor).
_PLAN_CACHE = {}


def _cached_block_plan(cu_seqlens, block_t):
    entry = _PLAN_CACHE.get("entry")
    if (
        entry is not None
        and entry["cu_seqlens"] is cu_seqlens
        and entry["block_t"] == block_t
    ):
        return entry["seq_of_block"], entry["base_of_block"]
    seq_of_block, base_of_block = _block_plan(cu_seqlens.tolist(), block_t)
    dev = cu_seqlens.device
    seq_t = torch.tensor(seq_of_block, dtype=torch.int32, device=dev)
    base_t = torch.tensor(base_of_block, dtype=torch.int32, device=dev)
    _PLAN_CACHE["entry"] = {
        "cu_seqlens": cu_seqlens,
        "block_t": block_t,
        "seq_of_block": seq_t,
        "base_of_block": base_t,
    }
    return seq_t, base_t


def kda_causal_conv1d_triton(
    x, weight, bias, conv_states, cu_seqlens, cache_indices,
    has_initial_state, pad_slot_id=-1,
):
    """Token-major causal conv with SiLU; returns the result in ``x``'s storage.

    Matches ``causal_conv1d_fwd(..., silu_activation=True, overwrite_x=True)``
    bit for bit and updates ``conv_states`` in place.
    """
    if not supports_kda_conv_triton(x, weight, conv_states):
        raise ValueError("unsupported Kimi-K3 KDA conv shape for the Triton path")
    T, C = x.shape
    if T == 0:
        return x
    seq_t, base_t = _cached_block_plan(cu_seqlens, _BLOCK_T)
    out = torch.empty_like(x)
    w = _weight_2d(weight).contiguous()
    b = (
        bias
        if bias is not None
        else torch.zeros(C, dtype=weight.dtype, device=x.device)
    )
    has_init = has_initial_state is not None
    init = has_initial_state.to(torch.int32) if has_init else cache_indices
    grid = (int(seq_t.numel()), triton.cdiv(C, _BLOCK_C))
    _kda_conv_kernel[grid](
        x, out, w, b, cu_seqlens, cache_indices, init, conv_states,
        seq_t, base_t,
        x.stride(0), out.stride(0),
        conv_states.stride(0), conv_states.stride(1), conv_states.stride(2),
        C, pad_slot_id,
        HAS_INIT=has_init, BLOCK_T=_BLOCK_T, BLOCK_C=_BLOCK_C,
        num_warps=4,
    )
    # Final state = last three RAW input rows, read before x is overwritten.
    nseq = int(cu_seqlens.numel()) - 1
    _kda_conv_state_kernel[(nseq, triton.cdiv(C, _BLOCK_C))](
        x, cu_seqlens, cache_indices, init, conv_states,
        x.stride(0),
        conv_states.stride(0), conv_states.stride(1), conv_states.stride(2),
        C, pad_slot_id,
        HAS_INIT=has_init, BLOCK_C=_BLOCK_C,
    )
    x.copy_(out)
    return x
