"""Model-specific MLA prefix-cache adapters.

The page lookup, GPU page materialization, and paged MLA extend prefill live in
generic prefix-cache helpers. This module keeps the remaining model glue in one
place: how each MLA model builds prefix extend contexts and projects attention
output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from batchgen.attention.mla.prefix_absorb import (
    build_absorbed_mla_query_states,
    prefix_rotary_seq_len,
    project_absorbed_mla_output,
    project_absorbed_mla_output_w8a16,
)

from .attention import AttnWrapperBase
from batchgen.kv_cache.prefill_offload import PrefillHostKVOffloader

from .prefix_cache import (
    PrefixCachePrepackMetadata,
    ensure_prefix_cache_prepack_metadata,
)
from .prefix_mla_extend import (
    MlaExtendSpec,
    run_prefix_mla_suffix_prefill_with_projected,
)

OutputProjector = Callable[[torch.Tensor], torch.Tensor]
ProjectedQueryBuilder = Callable[[object], torch.Tensor]


@dataclass(frozen=True)
class MlaPrefixBackendContext:
    """Prefix extend callbacks consumed by the existing MLA prepack backend."""

    wrapper: object
    metadata: PrefixCachePrepackMetadata
    spec: MlaExtendSpec
    suffix_query_builder: ProjectedQueryBuilder
    output_projection: OutputProjector
    prefill_prefix_materialization: object | None = None

    @property
    def prefix_reuse_mode(self) -> bool:
        return self.metadata.prefix_reuse_mode

    def rotary_seq_len(
        self,
        position_ids: torch.Tensor,
        fallback_seq_len: int,
    ) -> int:
        if self.metadata.full_seq_lengths:
            return prefix_rotary_seq_len(
                max(self.metadata.full_seq_lengths),
                position_ids,
            )
        return prefix_rotary_seq_len(fallback_seq_len, position_ids)

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
            prefill_prefix_materialization=self.prefill_prefix_materialization,
        )


def build_deepseek_prefix_backend_context(
    *,
    wrapper: object,
    metadata: PrefixCachePrepackMetadata,
) -> MlaPrefixBackendContext:
    metadata = ensure_prefix_cache_prepack_metadata(metadata)
    return _build_w8a16_prefix_backend_context(
        wrapper=wrapper,
        metadata=metadata,
        model_label="DeepSeek prefix extend",
        use_cached_absorb=False,
    )


def build_glm5_prefix_backend_context(
    *,
    wrapper: object,
    metadata: PrefixCachePrepackMetadata,
) -> MlaPrefixBackendContext:
    metadata = ensure_prefix_cache_prepack_metadata(metadata)
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
    metadata = ensure_prefix_cache_prepack_metadata(metadata)
    return MlaPrefixBackendContext(
        wrapper=wrapper,
        metadata=metadata,
        spec=_mla_extend_spec(wrapper),
        prefill_prefix_materialization=_prefill_prefix_materialization(wrapper),
        suffix_query_builder=lambda projection: build_absorbed_mla_query_states(
            q_nope=projection.q_nope,
            q_pe=projection.q_pe,
            dtype=projection.offload_kv.dtype,
            q_absorb=_kimi_q_absorb_weights(wrapper),
        ),
        output_projection=lambda attn_out: project_absorbed_mla_output(
            attn_out=attn_out,
            out_absorb=_kimi_out_absorb_weights(wrapper),
            v_head_dim=wrapper.module.v_head_dim,
            output_projection=wrapper.module.o_proj,
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
    offloader = PrefillHostKVOffloader(
        worker_view=worker_view,
        layer_idx=layer_idx,
        metadata=ensure_prefix_cache_prepack_metadata(metadata),
        track_task=AttnWrapperBase.track_prefill_offload_task,
        pin_tensor=AttnWrapperBase.pin_prefill_offload_tensor,
    )
    offloader.offload_mla(key=key)


def _mla_extend_spec(wrapper: object) -> MlaExtendSpec:
    attn = wrapper.module
    return MlaExtendSpec(
        kv_dim=attn.kv_lora_rank + attn.qk_rope_head_dim,
        num_heads=attn.num_heads,
        kv_lora_rank=attn.kv_lora_rank,
        softmax_scale=attn.softmax_scale,
    )


def _prefill_prefix_materialization(wrapper: object) -> object | None:
    return getattr(wrapper, "prefill_prefix_materialization", None)


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
        spec=_mla_extend_spec(wrapper),
        prefill_prefix_materialization=_prefill_prefix_materialization(wrapper),
        suffix_query_builder=lambda projection: build_absorbed_mla_query_states(
            q_nope=projection.q_nope,
            q_pe=projection.q_pe,
            dtype=projection.offload_kv.dtype,
            q_absorb=_w8a16_q_absorb_weights(
                wrapper,
                model_label=model_label,
                use_cached_absorb=use_cached_absorb,
            ),
        ),
        output_projection=lambda attn_out: _project_w8a16_absorbed_output(
            wrapper=wrapper,
            attn_out=attn_out,
            out_absorb=_w8a16_out_absorb_weights(
                wrapper,
                model_label=model_label,
                use_cached_absorb=use_cached_absorb,
            ),
            model_label=model_label,
        ),
    )


def _project_w8a16_absorbed_output(
    *,
    wrapper: object,
    attn_out: torch.Tensor,
    out_absorb: torch.Tensor,
    model_label: str,
) -> torch.Tensor:
    attn = wrapper.module
    from batchgen.attention.mla.fa3_backend import select_w8a16_gemm

    return project_absorbed_mla_output_w8a16(
        attn_out=attn_out,
        out_absorb=out_absorb,
        v_head_dim=attn.v_head_dim,
        o_proj_weight=attn.o_proj.weight.data,
        o_proj_scale=_weight_scale(
            wrapper, model_label, ("o_proj.weight_scale_inv",)
        )["o_proj.weight_scale_inv"],
        gemm=select_w8a16_gemm(),
    )


def _w8a16_q_absorb_weights(
    wrapper: object,
    *,
    model_label: str,
    use_cached_absorb: bool,
) -> torch.Tensor:
    if (
        use_cached_absorb
        and getattr(wrapper, "_cached_q_absorb", None) is not None
    ):
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

    from batchgen.attention.mla.flashmla_backend import (
        deepseek_v3_dequantization,
    )

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
