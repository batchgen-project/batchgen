"""Model-specific MLA prefix-cache adapters.

The page lookup, cached-prefix KV assembly, and FlashMLA replay live in the
generic prefix-cache helpers. This module keeps the remaining model glue in one
place: how each MLA model projects suffix/full-hit queries, builds suffix KV,
applies RoPE, and projects the replayed attention output.
"""

from __future__ import annotations

import os
from typing import Callable

import torch

from .attention import AttnWrapperBase
from .prefix_cache import (
    PrefixAwarePrefillOffloader,
    PrefixCachePrepackMetadata,
)
from .prefix_mla_replay import (
    MlaReplaySpec,
    run_prefix_mla_full_hit_prefill,
    run_prefix_mla_suffix_prefill,
)

SuffixProjector = Callable[
    [torch.Tensor, torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]
]
QueryProjector = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]
OutputProjector = Callable[[torch.Tensor], torch.Tensor]


def run_deepseek_prefix_aware_prefill(
    *,
    wrapper: object,
    hidden_states_2d: torch.Tensor,
    position_ids: torch.Tensor,
    metadata: PrefixCachePrepackMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run DeepSeek suffix prefill against cached prefix MLA KV."""
    return _run_mla_suffix_prefill(
        wrapper=wrapper,
        hidden_states_2d=hidden_states_2d,
        position_ids=position_ids,
        metadata=metadata,
        project_suffix_query_and_kv=lambda hidden, pos, full_len: (
            _project_w8a16_suffix_query_and_kv(
                wrapper=wrapper,
                hidden_states_2d=hidden,
                position_ids=pos,
                full_length=full_len,
                model_label="DeepSeek prefix replay",
                use_cached_absorb=False,
            )
        ),
        output_projection=lambda attn_out: _w8a16_output_projection(
            wrapper,
            attn_out,
            model_label="DeepSeek prefix replay",
            use_cached_absorb=False,
        ),
    )


def run_deepseek_full_hit_prefill(
    *,
    wrapper: object,
    hidden_states_2d: torch.Tensor,
    position_ids: torch.Tensor,
    metadata: PrefixCachePrepackMetadata,
) -> torch.Tensor:
    """Run DeepSeek exact full-hit prefill against fully cached MLA KV."""
    return _run_mla_full_hit_prefill(
        wrapper=wrapper,
        hidden_states_2d=hidden_states_2d,
        position_ids=position_ids,
        metadata=metadata,
        project_query=lambda hidden, pos, full_len: _project_w8a16_query_states(
            wrapper=wrapper,
            hidden_states_2d=hidden,
            position_ids=pos,
            full_length=full_len,
            model_label="DeepSeek prefix replay",
            use_cached_absorb=False,
        ),
        output_projection=lambda attn_out: _w8a16_output_projection(
            wrapper,
            attn_out,
            model_label="DeepSeek prefix replay",
            use_cached_absorb=False,
        ),
    )


def run_kimi_prefix_aware_prefill(
    *,
    wrapper: object,
    hidden_states_2d: torch.Tensor,
    position_ids: torch.Tensor,
    metadata: PrefixCachePrepackMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Kimi suffix prefill against cached prefix MLA KV."""
    return _run_mla_suffix_prefill(
        wrapper=wrapper,
        hidden_states_2d=hidden_states_2d,
        position_ids=position_ids,
        metadata=metadata,
        project_suffix_query_and_kv=lambda hidden, pos, full_len: (
            _project_kimi_suffix_query_and_kv(
                wrapper=wrapper,
                hidden_states_2d=hidden,
                position_ids=pos,
                full_length=full_len,
            )
        ),
        output_projection=lambda attn_out: _kimi_output_projection(
            wrapper,
            attn_out,
        ),
    )


def run_kimi_full_hit_prefill(
    *,
    wrapper: object,
    hidden_states_2d: torch.Tensor,
    position_ids: torch.Tensor,
    metadata: PrefixCachePrepackMetadata,
) -> torch.Tensor:
    """Run Kimi exact full-hit prefill against fully cached MLA KV."""
    return _run_mla_full_hit_prefill(
        wrapper=wrapper,
        hidden_states_2d=hidden_states_2d,
        position_ids=position_ids,
        metadata=metadata,
        project_query=lambda hidden, pos, full_len: _project_kimi_query_states(
            wrapper=wrapper,
            hidden_states_2d=hidden,
            position_ids=pos,
            full_length=full_len,
        ),
        output_projection=lambda attn_out: _kimi_output_projection(
            wrapper,
            attn_out,
        ),
    )


def run_glm5_prefix_aware_prefill(
    *,
    wrapper: object,
    hidden_states_2d: torch.Tensor,
    position_ids: torch.Tensor,
    metadata: PrefixCachePrepackMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run GLM-5 suffix prefill against cached prefix MLA KV."""
    if not metadata.prefix_reuse_mode:
        raise RuntimeError("GLM-5 prefix-aware prefill requires prefix reuse mode")
    if metadata.num_sequences != 1:
        raise RuntimeError(
            "GLM-5 prefix-aware prefill currently requires single-sequence "
            "micro-batches"
        )
    if metadata.prefix_shared_tokens is None or metadata.full_seq_lengths is None:
        raise RuntimeError("GLM-5 prefix-aware prefill requires prefix metadata")

    return _run_mla_suffix_prefill(
        wrapper=wrapper,
        hidden_states_2d=hidden_states_2d,
        position_ids=position_ids,
        metadata=metadata,
        project_suffix_query_and_kv=lambda hidden, pos, full_len: (
            _project_w8a16_suffix_query_and_kv(
                wrapper=wrapper,
                hidden_states_2d=hidden,
                position_ids=pos,
                full_length=full_len,
                model_label="GLM-5 prefix prefill",
                use_cached_absorb=True,
            )
        ),
        output_projection=lambda attn_out: _w8a16_output_projection(
            wrapper,
            attn_out,
            model_label="GLM-5 prefix prefill",
            use_cached_absorb=True,
        ),
    )


def run_glm5_full_hit_prefill(
    *,
    wrapper: object,
    hidden_states_2d: torch.Tensor,
    position_ids: torch.Tensor,
    metadata: PrefixCachePrepackMetadata,
) -> torch.Tensor:
    """Run GLM-5 exact full-hit prefill against fully cached MLA KV."""
    if not metadata.full_hit_mode:
        raise RuntimeError("GLM-5 full-hit prefill requires full-hit mode")
    if metadata.full_seq_lengths is None:
        raise RuntimeError("GLM-5 full-hit prefill requires full sequence lengths")
    metadata.validate_full_hit_query_lengths()

    return _run_mla_full_hit_prefill(
        wrapper=wrapper,
        hidden_states_2d=hidden_states_2d,
        position_ids=position_ids,
        metadata=metadata,
        project_query=lambda hidden, pos, full_len: _project_w8a16_query_states(
            wrapper=wrapper,
            hidden_states_2d=hidden,
            position_ids=pos,
            full_length=full_len,
            model_label="GLM-5 prefix prefill",
            use_cached_absorb=True,
        ),
        output_projection=lambda attn_out: _w8a16_output_projection(
            wrapper,
            attn_out,
            model_label="GLM-5 prefix prefill",
            use_cached_absorb=True,
        ),
    )


def offload_glm5_prepacked_mla_kv(
    *,
    key: torch.Tensor,
    worker_view: object,
    layer_idx: int,
    metadata: PrefixCachePrepackMetadata,
) -> None:
    """Offload prepacked GLM-5 k-only MLA/indexer KV with prefix offsets."""
    offloader = PrefixAwarePrefillOffloader(
        worker_view=worker_view,
        layer_idx=layer_idx,
        metadata=metadata,
        track_task=AttnWrapperBase.track_prefill_offload_task,
        pin_tensor=AttnWrapperBase.pin_prefill_offload_tensor,
    )
    offloader.offload_mla(key=key)


def _run_mla_suffix_prefill(
    *,
    wrapper: object,
    hidden_states_2d: torch.Tensor,
    position_ids: torch.Tensor,
    metadata: PrefixCachePrepackMetadata,
    project_suffix_query_and_kv: SuffixProjector,
    output_projection: OutputProjector,
) -> tuple[torch.Tensor, torch.Tensor]:
    return run_prefix_mla_suffix_prefill(
        wrapper=wrapper,
        hidden_states_2d=hidden_states_2d,
        position_ids=position_ids,
        metadata=metadata,
        spec=_mla_replay_spec(wrapper),
        project_suffix_query_and_kv=project_suffix_query_and_kv,
        output_projection=output_projection,
    )


def _run_mla_full_hit_prefill(
    *,
    wrapper: object,
    hidden_states_2d: torch.Tensor,
    position_ids: torch.Tensor,
    metadata: PrefixCachePrepackMetadata,
    project_query: QueryProjector,
    output_projection: OutputProjector,
) -> torch.Tensor:
    return run_prefix_mla_full_hit_prefill(
        wrapper=wrapper,
        hidden_states_2d=hidden_states_2d,
        position_ids=position_ids,
        metadata=metadata,
        spec=_mla_replay_spec(wrapper),
        project_query=project_query,
        output_projection=output_projection,
    )


def _mla_replay_spec(wrapper: object) -> MlaReplaySpec:
    attn = wrapper.module
    return MlaReplaySpec(
        kv_dim=attn.kv_lora_rank + attn.qk_rope_head_dim,
        num_heads=attn.num_heads,
        kv_lora_rank=attn.kv_lora_rank,
        softmax_scale=attn.softmax_scale,
    )


def _project_w8a16_suffix_query_and_kv(
    *,
    wrapper: object,
    hidden_states_2d: torch.Tensor,
    position_ids: torch.Tensor,
    full_length: int,
    model_label: str,
    use_cached_absorb: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    attn = wrapper.module
    weight_scale = _weight_scale(
        wrapper,
        model_label,
        (
            "q_a_proj.weight_scale_inv",
            "q_b_proj.weight_scale_inv",
            "kv_a_proj_with_mqa.weight_scale_inv",
        ),
    )

    q_states = _w8a16_gemm(
        attn.q_a_proj.weight.data,
        weight_scale["q_a_proj.weight_scale_inv"],
        hidden_states_2d,
    )
    q_states = attn.q_a_layernorm(q_states)
    q_states = _w8a16_gemm(
        attn.q_b_proj.weight.data,
        weight_scale["q_b_proj.weight_scale_inv"],
        q_states,
    )
    total_tokens = hidden_states_2d.shape[0]
    q_states = q_states.view(total_tokens, attn.num_heads, attn.q_head_dim)
    q_nope, q_pe = torch.split(
        q_states,
        [attn.qk_nope_head_dim, attn.qk_rope_head_dim],
        dim=-1,
    )

    compressed_kv = _w8a16_gemm(
        attn.kv_a_proj_with_mqa.weight.data,
        weight_scale["kv_a_proj_with_mqa.weight_scale_inv"],
        hidden_states_2d,
    )
    kv, k_pe = torch.split(
        compressed_kv,
        [attn.kv_lora_rank, attn.qk_rope_head_dim],
        dim=-1,
    )
    kv = attn.kv_a_layernorm(kv)
    k_pe = k_pe.view(total_tokens, 1, attn.qk_rope_head_dim)

    q_pe, k_pe = _apply_interleaved_rope(
        attn=attn,
        q_pe=q_pe,
        k_pe=k_pe,
        position_ids=position_ids,
        full_length=full_length,
    )
    if k_pe is None:
        raise RuntimeError(f"{model_label} failed to build suffix k_pe")
    offload_kv = torch.cat(
        [kv, k_pe.view(total_tokens, attn.qk_rope_head_dim)],
        dim=-1,
    )
    q_absorb = _w8a16_q_absorb_weights(
        wrapper,
        model_label=model_label,
        use_cached_absorb=use_cached_absorb,
    )
    return (
        _absorbed_query_states(
            wrapper,
            q_nope,
            q_pe,
            offload_kv.dtype,
            q_absorb=q_absorb,
        ),
        offload_kv,
    )


def _project_w8a16_query_states(
    *,
    wrapper: object,
    hidden_states_2d: torch.Tensor,
    position_ids: torch.Tensor,
    full_length: int,
    model_label: str,
    use_cached_absorb: bool,
) -> torch.Tensor:
    attn = wrapper.module
    weight_scale = _weight_scale(
        wrapper,
        model_label,
        ("q_a_proj.weight_scale_inv", "q_b_proj.weight_scale_inv"),
    )
    q_states = _w8a16_gemm(
        attn.q_a_proj.weight.data,
        weight_scale["q_a_proj.weight_scale_inv"],
        hidden_states_2d,
    )
    q_states = attn.q_a_layernorm(q_states)
    q_states = _w8a16_gemm(
        attn.q_b_proj.weight.data,
        weight_scale["q_b_proj.weight_scale_inv"],
        q_states,
    )
    total_tokens = hidden_states_2d.shape[0]
    q_states = q_states.view(total_tokens, attn.num_heads, attn.q_head_dim)
    q_nope, q_pe = torch.split(
        q_states,
        [attn.qk_nope_head_dim, attn.qk_rope_head_dim],
        dim=-1,
    )
    q_pe, _ = _apply_interleaved_rope(
        attn=attn,
        q_pe=q_pe,
        k_pe=None,
        position_ids=position_ids,
        full_length=full_length,
    )
    q_absorb = _w8a16_q_absorb_weights(
        wrapper,
        model_label=model_label,
        use_cached_absorb=use_cached_absorb,
    )
    return _absorbed_query_states(
        wrapper,
        q_nope,
        q_pe,
        q_pe.dtype,
        q_absorb=q_absorb,
    ).view(
        total_tokens,
        1,
        attn.num_heads,
        attn.kv_lora_rank + attn.qk_rope_head_dim,
    ).contiguous()


def _project_kimi_suffix_query_and_kv(
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

    q_pe, k_pe = _apply_standard_rope(
        attn=attn,
        q_pe=q_pe,
        k_pe=k_pe,
        position_ids=position_ids,
        full_length=full_length,
    )
    if k_pe is None:
        raise RuntimeError("Kimi prefix replay failed to build suffix k_pe")
    offload_kv = torch.cat(
        [kv, k_pe.view(total_tokens, attn.qk_rope_head_dim)],
        dim=-1,
    )
    return (
        _absorbed_query_states(
            wrapper,
            q_nope,
            q_pe,
            offload_kv.dtype,
            q_absorb=_kimi_q_absorb_weights(wrapper),
        ),
        offload_kv,
    )


def _project_kimi_query_states(
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
    q_pe, _ = _apply_standard_rope(
        attn=attn,
        q_pe=q_pe,
        k_pe=None,
        position_ids=position_ids,
        full_length=full_length,
    )
    return _absorbed_query_states(
        wrapper,
        q_nope,
        q_pe,
        q_pe.dtype,
        q_absorb=_kimi_q_absorb_weights(wrapper),
    ).view(
        total_tokens,
        1,
        attn.num_heads,
        attn.kv_lora_rank + attn.qk_rope_head_dim,
    ).contiguous()


def _apply_interleaved_rope(
    *,
    attn: object,
    q_pe: torch.Tensor,
    k_pe: torch.Tensor | None,
    position_ids: torch.Tensor,
    full_length: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    from batchgen.attention.mla.rotary_embedding import (
        rotary_pos_emb_interleaved_native,
    )

    rotary_seq_len = max(int(full_length), int(position_ids.max().item()) + 1)
    cos, sin = attn.rotary_emb(q_pe.unsqueeze(0), seq_len=rotary_seq_len)
    q_pe = rotary_pos_emb_interleaved_native(
        q_pe.unsqueeze(0),
        cos,
        sin,
        position_ids.unsqueeze(0),
        2,
    ).squeeze(0)
    if k_pe is None:
        return q_pe, None
    k_pe = rotary_pos_emb_interleaved_native(
        k_pe.unsqueeze(0),
        cos,
        sin,
        position_ids.unsqueeze(0),
        2,
    ).squeeze(0)
    return q_pe, k_pe


def _apply_standard_rope(
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
    *,
    q_absorb: torch.Tensor,
) -> torch.Tensor:
    attn = wrapper.module
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


def _w8a16_output_projection(
    wrapper: object,
    attn_out: torch.Tensor,
    *,
    model_label: str,
    use_cached_absorb: bool,
) -> torch.Tensor:
    attn = wrapper.module
    out_absorb = _w8a16_out_absorb_weights(
        wrapper,
        model_label=model_label,
        use_cached_absorb=use_cached_absorb,
    )
    attn_output = torch.einsum("bqhc,hdc->bqhd", attn_out, out_absorb)
    attn_output = attn_output.reshape(
        attn_out.shape[0] * attn_out.shape[1],
        attn.num_heads * attn.v_head_dim,
    )
    return _w8a16_gemm(
        attn.o_proj.weight.data,
        _weight_scale(wrapper, model_label, ("o_proj.weight_scale_inv",))[
            "o_proj.weight_scale_inv"
        ],
        attn_output,
    )


def _kimi_output_projection(wrapper: object, attn_out: torch.Tensor) -> torch.Tensor:
    attn = wrapper.module
    out_absorb = _kimi_out_absorb_weights(wrapper)
    attn_output = torch.einsum("bqhc,hdc->bqhd", attn_out, out_absorb)
    attn_output = attn_output.reshape(
        attn_out.shape[0] * attn_out.shape[1],
        attn.num_heads * attn.v_head_dim,
    )
    return attn.o_proj(attn_output)


def _w8a16_q_absorb_weights(
    wrapper: object,
    *,
    model_label: str,
    use_cached_absorb: bool,
) -> torch.Tensor:
    if use_cached_absorb and getattr(wrapper, "_cached_q_absorb", None) is not None:
        return wrapper._cached_q_absorb
    attn = wrapper.module
    if use_cached_absorb and getattr(attn, "q_absorb", None) is not None:
        return attn.q_absorb
    kv_b_proj = _dequantized_kv_b_proj(wrapper, model_label)
    return kv_b_proj[:, : attn.qk_nope_head_dim, :]


def _w8a16_out_absorb_weights(
    wrapper: object,
    *,
    model_label: str,
    use_cached_absorb: bool,
) -> torch.Tensor:
    if (
        use_cached_absorb
        and getattr(wrapper, "_cached_out_absorb", None) is not None
    ):
        return wrapper._cached_out_absorb
    attn = wrapper.module
    if use_cached_absorb and getattr(attn, "out_absorb", None) is not None:
        return attn.out_absorb
    kv_b_proj = _dequantized_kv_b_proj(wrapper, model_label)
    return kv_b_proj[:, attn.qk_nope_head_dim :, :]


def _dequantized_kv_b_proj(wrapper: object, model_label: str) -> torch.Tensor:
    attn = wrapper.module
    weight_scale = _weight_scale(
        wrapper,
        model_label,
        ("kv_b_proj.weight_scale_inv",),
    )

    from batchgen.attention.mla.flashmla_backend import deepseek_v3_dequantization

    return deepseek_v3_dequantization(
        attn.kv_b_proj.weight.data,
        weight_scale["kv_b_proj.weight_scale_inv"],
    ).view(
        attn.num_heads,
        -1,
        attn.kv_lora_rank,
    )


def _kimi_q_absorb_weights(wrapper: object) -> torch.Tensor:
    attn = wrapper.module
    if getattr(attn, "q_absorb", None) is not None:
        return attn.q_absorb
    return _kimi_kv_b_proj(wrapper)[:, : attn.qk_nope_head_dim, :]


def _kimi_out_absorb_weights(wrapper: object) -> torch.Tensor:
    attn = wrapper.module
    if getattr(attn, "out_absorb", None) is not None:
        return attn.out_absorb
    return _kimi_kv_b_proj(wrapper)[:, attn.qk_nope_head_dim :, :]


def _kimi_kv_b_proj(wrapper: object) -> torch.Tensor:
    attn = wrapper.module
    return attn.kv_b_proj.weight.data.view(
        attn.num_heads,
        -1,
        attn.kv_lora_rank,
    )


def _weight_scale(
    wrapper: object,
    model_label: str,
    required_keys: tuple[str, ...],
) -> dict:
    weight_scale = getattr(wrapper, "weight_dequant_scale", None)
    missing = [
        key
        for key in required_keys
        if weight_scale is None or key not in weight_scale
    ]
    if missing:
        raise RuntimeError(
            f"{model_label} requires weight scales: {', '.join(missing)}"
        )
    return weight_scale


def _w8a16_gemm(
    weight_data_fp8: torch.Tensor,
    weight_scale_inv_fp32: torch.Tensor,
    activation_bf16: torch.Tensor,
) -> torch.Tensor:
    from batchgen.attention.mla.fa3_backend import (
        w8a16_gemm,
        w8a16_gemm_dequant,
    )

    use_dequant_path = os.environ.get("BATCHGEN_W8A16_DEQUANT", "0") == "1"
    gemm = w8a16_gemm_dequant if use_dequant_path else w8a16_gemm
    return gemm(weight_data_fp8, weight_scale_inv_fp32, activation_bf16)
