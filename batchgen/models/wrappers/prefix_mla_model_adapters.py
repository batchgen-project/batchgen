"""Model-specific MLA prefix-cache adapters.

The page lookup, cached-prefix KV assembly, and FlashMLA replay live in the
generic prefix-cache helpers. This module keeps the remaining model glue in one
place: how each MLA model builds prefix replay contexts and projects the replayed
attention output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .attention import AttnWrapperBase
from .prefix_cache import (
    PrefixAwarePrefillOffloader,
    PrefixCachePrepackMetadata,
)
from .prefix_mla_replay import (
    MlaReplaySpec,
    run_prefix_mla_full_hit_prefill_with_query,
    run_prefix_mla_suffix_prefill_with_projected,
)

OutputProjector = Callable[[torch.Tensor], torch.Tensor]
ProjectedQueryBuilder = Callable[[object], torch.Tensor]


@dataclass(frozen=True)
class MlaPrefixBackendContext:
    """Prefix replay callbacks consumed by the existing MLA prepack backend."""

    wrapper: object
    metadata: PrefixCachePrepackMetadata
    spec: MlaReplaySpec
    suffix_query_builder: ProjectedQueryBuilder
    full_hit_query_builder: ProjectedQueryBuilder
    output_projection: OutputProjector

    @property
    def prefix_reuse_mode(self) -> bool:
        return self.metadata.prefix_reuse_mode

    @property
    def full_hit_mode(self) -> bool:
        return self.metadata.full_hit_mode

    def rotary_seq_len(
        self,
        position_ids: torch.Tensor,
        fallback_seq_len: int,
    ) -> int:
        if self.metadata.full_seq_lengths:
            return _rotary_seq_len(max(self.metadata.full_seq_lengths), position_ids)
        return _rotary_seq_len(fallback_seq_len, position_ids)

    def run_suffix_prefill(
        self,
        projection: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        offload_kv = getattr(projection, "offload_kv", None)
        if offload_kv is None:
            raise RuntimeError("MLA prefix backend context requires suffix KV")
        return run_prefix_mla_suffix_prefill_with_projected(
            wrapper=self.wrapper,
            query_states=self.suffix_query_builder(projection),
            offload_kv=offload_kv,
            metadata=self.metadata,
            spec=self.spec,
            output_projection=self.output_projection,
        )

    def run_full_hit_prefill(self, projection: object) -> torch.Tensor:
        return run_prefix_mla_full_hit_prefill_with_query(
            wrapper=self.wrapper,
            query_states=self.full_hit_query_builder(projection),
            metadata=self.metadata,
            spec=self.spec,
            output_projection=self.output_projection,
        )


def build_deepseek_prefix_backend_context(
    *,
    wrapper: object,
    metadata: PrefixCachePrepackMetadata,
) -> MlaPrefixBackendContext:
    return _build_w8a16_prefix_backend_context(
        wrapper=wrapper,
        metadata=metadata,
        model_label="DeepSeek prefix replay",
        use_cached_absorb=False,
    )


def build_glm5_prefix_backend_context(
    *,
    wrapper: object,
    metadata: PrefixCachePrepackMetadata,
) -> MlaPrefixBackendContext:
    return _build_w8a16_prefix_backend_context(
        wrapper=wrapper,
        metadata=metadata,
        model_label="GLM-5 prefix prefill",
        use_cached_absorb=True,
    )


def build_kimi_prefix_backend_context(
    *,
    wrapper: object,
    metadata: PrefixCachePrepackMetadata,
) -> MlaPrefixBackendContext:
    return MlaPrefixBackendContext(
        wrapper=wrapper,
        metadata=metadata,
        spec=_mla_replay_spec(wrapper),
        suffix_query_builder=lambda projection: _absorbed_query_states(
            wrapper,
            projection.q_nope,
            projection.q_pe,
            projection.offload_kv.dtype,
            q_absorb=_kimi_q_absorb_weights(wrapper),
        ),
        full_hit_query_builder=lambda projection: _full_hit_query_from_projection(
            wrapper,
            projection,
            projection.q_pe.dtype,
            q_absorb=_kimi_q_absorb_weights(wrapper),
        ),
        output_projection=lambda attn_out: _kimi_output_projection(wrapper, attn_out),
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


def _mla_replay_spec(wrapper: object) -> MlaReplaySpec:
    attn = wrapper.module
    return MlaReplaySpec(
        kv_dim=attn.kv_lora_rank + attn.qk_rope_head_dim,
        num_heads=attn.num_heads,
        kv_lora_rank=attn.kv_lora_rank,
        softmax_scale=attn.softmax_scale,
    )


def _build_w8a16_prefix_backend_context(
    *,
    wrapper: object,
    metadata: PrefixCachePrepackMetadata,
    model_label: str,
    use_cached_absorb: bool,
) -> MlaPrefixBackendContext:
    return MlaPrefixBackendContext(
        wrapper=wrapper,
        metadata=metadata,
        spec=_mla_replay_spec(wrapper),
        suffix_query_builder=lambda projection: _absorbed_query_states(
            wrapper,
            projection.q_nope,
            projection.q_pe,
            projection.offload_kv.dtype,
            q_absorb=_w8a16_q_absorb_weights(
                wrapper,
                model_label=model_label,
                use_cached_absorb=use_cached_absorb,
            ),
        ),
        full_hit_query_builder=lambda projection: _full_hit_query_from_projection(
            wrapper,
            projection,
            projection.q_pe.dtype,
            q_absorb=_w8a16_q_absorb_weights(
                wrapper,
                model_label=model_label,
                use_cached_absorb=use_cached_absorb,
            ),
        ),
        output_projection=lambda attn_out: _w8a16_output_projection(
            wrapper,
            attn_out,
            model_label=model_label,
            use_cached_absorb=use_cached_absorb,
        ),
    )


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


def _full_hit_query_from_projection(
    wrapper: object,
    projection: object,
    dtype: torch.dtype,
    *,
    q_absorb: torch.Tensor,
) -> torch.Tensor:
    attn = wrapper.module
    total_tokens = projection.q_nope.shape[0]
    return _absorbed_query_states(
        wrapper,
        projection.q_nope,
        projection.q_pe,
        dtype,
        q_absorb=q_absorb,
    ).view(
        total_tokens,
        1,
        attn.num_heads,
        attn.kv_lora_rank + attn.qk_rope_head_dim,
    ).contiguous()


def _rotary_seq_len(full_length: int, position_ids: torch.Tensor) -> int:
    return max(int(full_length), int(position_ids.max().item()) + 1)


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
    from batchgen.attention.mla.fa3_backend import select_w8a16_gemm
    return select_w8a16_gemm()(
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
