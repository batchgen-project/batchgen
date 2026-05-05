"""Kimi K2.5 MLA prefix-cache replay helpers."""

from __future__ import annotations

import torch

from batchgen.models.wrappers.prefix_cache import PrefixCachePrepackMetadata
from batchgen.models.wrappers.prefix_mla_replay import (
    MlaReplaySpec,
    run_prefix_mla_full_hit_prefill,
    run_prefix_mla_suffix_prefill,
)


def run_kimi_prefix_aware_prefill(
    *,
    wrapper: object,
    hidden_states_2d: torch.Tensor,
    position_ids: torch.Tensor,
    metadata: PrefixCachePrepackMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Kimi suffix prefill against cached prefix MLA KV."""
    return run_prefix_mla_suffix_prefill(
        wrapper=wrapper,
        hidden_states_2d=hidden_states_2d,
        position_ids=position_ids,
        metadata=metadata,
        spec=_mla_replay_spec(wrapper),
        project_suffix_query_and_kv=lambda hidden, pos, full_len: (
            _project_suffix_query_and_kv(
                wrapper=wrapper,
                hidden_states_2d=hidden,
                position_ids=pos,
                full_length=full_len,
            )
        ),
        output_projection=lambda attn_out: _output_projection(wrapper, attn_out),
    )


def run_kimi_full_hit_prefill(
    *,
    wrapper: object,
    hidden_states_2d: torch.Tensor,
    position_ids: torch.Tensor,
    metadata: PrefixCachePrepackMetadata,
) -> torch.Tensor:
    """Run Kimi exact full-hit prefill against fully cached MLA KV."""
    return run_prefix_mla_full_hit_prefill(
        wrapper=wrapper,
        hidden_states_2d=hidden_states_2d,
        position_ids=position_ids,
        metadata=metadata,
        spec=_mla_replay_spec(wrapper),
        project_query=lambda hidden, pos, full_len: _project_query_states(
            wrapper=wrapper,
            hidden_states_2d=hidden,
            position_ids=pos,
            full_length=full_len,
        ),
        output_projection=lambda attn_out: _output_projection(wrapper, attn_out),
    )


def _mla_replay_spec(wrapper: object) -> MlaReplaySpec:
    attn = wrapper.module
    return MlaReplaySpec(
        kv_dim=attn.kv_lora_rank + attn.qk_rope_head_dim,
        num_heads=attn.num_heads,
        kv_lora_rank=attn.kv_lora_rank,
        softmax_scale=attn.softmax_scale,
    )


def _project_suffix_query_and_kv(
    *,
    wrapper: object,
    hidden_states_2d: torch.Tensor,
    position_ids: torch.Tensor,
    full_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    attn = wrapper.module
    q_states = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(hidden_states_2d)))
    total_tokens = hidden_states_2d.shape[0]
    q_states = q_states.view(total_tokens, attn.num_heads, attn.q_head_dim)
    q_nope, q_pe = torch.split(
        q_states,
        [attn.qk_nope_head_dim, attn.qk_rope_head_dim],
        dim=-1,
    )

    compressed_kv = attn.kv_a_proj_with_mqa(hidden_states_2d)
    kv, k_pe = torch.split(
        compressed_kv,
        [attn.kv_lora_rank, attn.qk_rope_head_dim],
        dim=-1,
    )
    kv = attn.kv_a_layernorm(kv)
    k_pe = k_pe.view(total_tokens, 1, attn.qk_rope_head_dim)

    q_pe, k_pe = _apply_rope(
        attn=attn,
        q_pe=q_pe,
        k_pe=k_pe,
        position_ids=position_ids,
        full_length=full_length,
    )
    offload_kv = torch.cat(
        [kv, k_pe.view(total_tokens, attn.qk_rope_head_dim)],
        dim=-1,
    )
    return _absorbed_query_states(wrapper, q_nope, q_pe, offload_kv.dtype), offload_kv


def _project_query_states(
    *,
    wrapper: object,
    hidden_states_2d: torch.Tensor,
    position_ids: torch.Tensor,
    full_length: int,
) -> torch.Tensor:
    attn = wrapper.module
    q_states = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(hidden_states_2d)))
    total_tokens = hidden_states_2d.shape[0]
    q_states = q_states.view(total_tokens, attn.num_heads, attn.q_head_dim)
    q_nope, q_pe = torch.split(
        q_states,
        [attn.qk_nope_head_dim, attn.qk_rope_head_dim],
        dim=-1,
    )
    q_pe, _ = _apply_rope(
        attn=attn,
        q_pe=q_pe,
        k_pe=None,
        position_ids=position_ids,
        full_length=full_length,
    )
    return _absorbed_query_states(wrapper, q_nope, q_pe, q_pe.dtype).view(
        total_tokens,
        1,
        attn.num_heads,
        attn.kv_lora_rank + attn.qk_rope_head_dim,
    ).contiguous()


def _apply_rope(
    *,
    attn: object,
    q_pe: torch.Tensor,
    k_pe: torch.Tensor | None,
    position_ids: torch.Tensor,
    full_length: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    from batchgen.attention.mla.rotary_embedding import rotary_pos_emb

    rotary_seq_len = max(int(full_length), int(position_ids.max().item()) + 1)
    cos, sin = attn.rotary_emb(q_pe.unsqueeze(0), seq_len=rotary_seq_len)
    q_pe = rotary_pos_emb(
        q_pe.unsqueeze(0),
        cos,
        sin,
        position_ids.unsqueeze(0),
        2,
    ).squeeze(0)
    if k_pe is None:
        return q_pe, None
    k_pe = rotary_pos_emb(
        k_pe.unsqueeze(0),
        cos,
        sin,
        position_ids.unsqueeze(0),
        2,
    ).squeeze(0)
    return q_pe, k_pe


def _absorbed_query_states(
    wrapper: object,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    attn = wrapper.module
    q_absorb = _q_absorb_weights(wrapper)
    total_tokens = q_nope.shape[0]
    query_states = torch.empty(
        1,
        total_tokens,
        attn.num_heads,
        attn.kv_lora_rank + attn.qk_rope_head_dim,
        dtype=dtype,
        device=q_pe.device,
    )
    query_states[0, :, :, : attn.kv_lora_rank] = torch.einsum(
        "thd,hdc->thc",
        q_nope,
        q_absorb,
    )
    query_states[0, :, :, attn.kv_lora_rank :] = q_pe
    return query_states.contiguous()


def _output_projection(wrapper: object, attn_out: torch.Tensor) -> torch.Tensor:
    attn = wrapper.module
    out_absorb = _out_absorb_weights(wrapper)
    attn_output = torch.einsum("bqhc,hdc->bqhd", attn_out, out_absorb)
    attn_output = attn_output.reshape(
        attn_out.shape[0] * attn_out.shape[1],
        attn.num_heads * attn.v_head_dim,
    )
    return attn.o_proj(attn_output)


def _q_absorb_weights(wrapper: object) -> torch.Tensor:
    attn = wrapper.module
    if getattr(attn, "q_absorb", None) is not None:
        return attn.q_absorb
    return _kv_b_proj(wrapper)[:, : attn.qk_nope_head_dim, :]


def _out_absorb_weights(wrapper: object) -> torch.Tensor:
    attn = wrapper.module
    if getattr(attn, "out_absorb", None) is not None:
        return attn.out_absorb
    return _kv_b_proj(wrapper)[:, attn.qk_nope_head_dim :, :]


def _kv_b_proj(wrapper: object) -> torch.Tensor:
    attn = wrapper.module
    return attn.kv_b_proj.weight.data.view(
        attn.num_heads,
        -1,
        attn.kv_lora_rank,
    )
