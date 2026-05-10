"""MLA absorb helpers used by prefix-cache prefill paths."""

from __future__ import annotations

from typing import Callable

import torch

W8A16GemmFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def prefix_rotary_seq_len(full_length: int, position_ids: torch.Tensor) -> int:
    """Return the RoPE seq-len needed for a prefix-aware prefill batch."""

    return max(int(full_length), int(position_ids.max().item()) + 1)


def build_absorbed_mla_query_states(
    *,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    q_absorb: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build FlashMLA query states from projected MLA q_nope/q_pe tensors."""

    total_tokens = q_nope.shape[0]
    num_heads = q_nope.shape[1]
    kv_lora_rank = q_absorb.shape[2]
    query_states = torch.empty(
        1,
        total_tokens,
        num_heads,
        kv_lora_rank + q_pe.shape[-1],
        dtype=dtype,
        device=q_pe.device,
    )
    query_states[0, :, :, :kv_lora_rank] = torch.einsum(
        "thd,hdc->thc",
        q_nope,
        q_absorb,
    )
    query_states[0, :, :, kv_lora_rank:] = q_pe
    return query_states.contiguous()


def build_full_hit_absorbed_mla_query_states(
    *,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    q_absorb: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build full-hit query states in the shape expected by prefix replay."""

    query_states = build_absorbed_mla_query_states(
        q_nope=q_nope,
        q_pe=q_pe,
        q_absorb=q_absorb,
        dtype=dtype,
    )
    return query_states.view(
        q_nope.shape[0],
        1,
        q_nope.shape[1],
        query_states.shape[-1],
    ).contiguous()


def absorb_mla_attention_output(
    *,
    attn_out: torch.Tensor,
    out_absorb: torch.Tensor,
    v_head_dim: int,
) -> torch.Tensor:
    """Apply MLA out-absorb and flatten heads for the final output projection."""

    attn_output = torch.einsum("bqhc,hdc->bqhd", attn_out, out_absorb)
    return attn_output.reshape(
        attn_out.shape[0] * attn_out.shape[1],
        attn_out.shape[2] * int(v_head_dim),
    )


def project_absorbed_mla_output(
    *,
    attn_out: torch.Tensor,
    out_absorb: torch.Tensor,
    v_head_dim: int,
    output_projection: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Apply out-absorb followed by a BF16/regular output projection."""

    return output_projection(
        absorb_mla_attention_output(
            attn_out=attn_out,
            out_absorb=out_absorb,
            v_head_dim=v_head_dim,
        )
    )


def project_absorbed_mla_output_w8a16(
    *,
    attn_out: torch.Tensor,
    out_absorb: torch.Tensor,
    v_head_dim: int,
    o_proj_weight: torch.Tensor,
    o_proj_scale: torch.Tensor,
    gemm: W8A16GemmFn,
) -> torch.Tensor:
    """Apply out-absorb followed by the selected W8A16 output GEMM."""

    return gemm(
        o_proj_weight,
        o_proj_scale,
        absorb_mla_attention_output(
            attn_out=attn_out,
            out_absorb=out_absorb,
            v_head_dim=v_head_dim,
        ),
    )
