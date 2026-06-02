"""DeepSeek-V4 runtime per-layer attention dispatcher.

Ported (slim) from sglang `deepseek_v4_backend.py`. The upstream class is 1255
LOC and tightly coupled to sglang's `AttentionBackend` + `ForwardBatch`. This
port keeps the *dispatch contract* (per-layer path selection driven by
`compress_ratio` + `c4_sparse_topk`) without dragging in those base classes.

Model code constructs a `DSV4AttnMetadata` for the current step, the dispatcher
selects one of three paths per layer, and calls the matching batchgen kernel.

Three attention paths (matches upstream):
    - dense MLA decode             (compress_ratio == 0)
    - c4 sparse (indexer top-512)  (compress_ratio == 4)
    - c128 compressed (HCA)        (compress_ratio == 128)

The actual kernel calls go through:
    - batchgen.attention.mla.flashmla_backend (FlashMLA dense / sparse decode)
    - batchgen_kernels.attention.dsa.fused_indexer_score (C4 indexer pipeline)
    - batchgen_kernels.attention.v4_compressor (C128 compressor)
    - batchgen.attention.dsa.v4_indexer_metadata (per-step metadata init)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

SWA_WINDOW = 128
C4_TOPK = 512
PAGE_INDEX_ALIGNED_SIZE = 64


class V4AttnPath(enum.Enum):
    DENSE_MLA = "dense_mla"
    C4_SPARSE = "c4_sparse"
    C128_COMPRESS = "c128_compress"

    @classmethod
    def from_compress_ratio(cls, compress_ratio: int) -> "V4AttnPath":
        if compress_ratio == 0:
            return cls.DENSE_MLA
        if compress_ratio == 4:
            return cls.C4_SPARSE
        if compress_ratio == 128:
            return cls.C128_COMPRESS
        raise ValueError(
            f"unsupported compress_ratio={compress_ratio}; "
            f"expected one of {{0, 4, 128}}"
        )


@dataclass
class DSV4AttnMetadata:
    """Per-forward-step metadata, constructed once and read by every layer.

    Matches the upstream `DSV4AttnMetadata` field set, minus the fields that
    require sglang's `FlashMLASchedMeta` (we initialize FlashMLA metadata
    on-demand inside the dispatcher's call sites).
    """

    page_size: int
    page_table: torch.Tensor
    raw_out_loc: torch.Tensor
    seq_lens_casual: torch.Tensor
    positions_casual: torch.Tensor

    swa_page_indices: torch.Tensor
    swa_topk_lengths: torch.Tensor

    c4_sparse_topk: int = C4_TOPK
    c4_out_loc: Optional[torch.Tensor] = None
    c4_topk_lengths_raw: Optional[torch.Tensor] = None
    c4_topk_lengths_clamp1: Optional[torch.Tensor] = None

    c128_out_loc: Optional[torch.Tensor] = None
    c128_page_indices: Optional[torch.Tensor] = None
    c128_topk_lengths_clamp1: Optional[torch.Tensor] = None

    extras: dict = field(default_factory=dict)


@dataclass
class DSV4LayerConfig:
    layer_idx: int
    compress_ratio: int
    n_heads: int
    head_dim: int
    rope_head_dim: int
    swa_window: int = SWA_WINDOW

    @property
    def path(self) -> V4AttnPath:
        return V4AttnPath.from_compress_ratio(self.compress_ratio)


class DeepseekV4AttnBackend:
    """Per-layer attention dispatcher.

    Construct once per model. Call ``init_metadata(...)`` at the start of every
    forward pass to populate the per-step ``DSV4AttnMetadata``. Layer modules
    then call ``forward(layer_config, q, kv, ...)`` which routes to the right
    kernel.
    """

    def __init__(
        self,
        layer_configs: list[DSV4LayerConfig],
        page_size: int = 64,
        flashmla_backend: Any = None,
    ):
        self.layer_configs = layer_configs
        self.page_size = page_size
        self._flashmla = flashmla_backend
        self._metadata: Optional[DSV4AttnMetadata] = None
        self._fused_indexer = None
        self._compressor = None

    def init_metadata(self, metadata: DSV4AttnMetadata) -> None:
        self._metadata = metadata

    def clear_metadata(self) -> None:
        self._metadata = None

    @property
    def metadata(self) -> DSV4AttnMetadata:
        if self._metadata is None:
            raise RuntimeError(
                "DeepseekV4AttnBackend.metadata accessed before init_metadata()"
            )
        return self._metadata

    def forward(
        self,
        layer_config: DSV4LayerConfig,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        path = layer_config.path
        if path is V4AttnPath.DENSE_MLA:
            return self._forward_dense_mla(
                layer_config, q, kv, attn_sink, **kwargs
            )
        if path is V4AttnPath.C4_SPARSE:
            return self._forward_c4_sparse(
                layer_config, q, kv, attn_sink, **kwargs
            )
        if path is V4AttnPath.C128_COMPRESS:
            return self._forward_c128_compress(
                layer_config, q, kv, attn_sink, **kwargs
            )
        raise AssertionError(f"unreachable path={path}")

    def _forward_dense_mla(
        self,
        layer_config: DSV4LayerConfig,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: Optional[torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        if self._flashmla is None:
            raise NotImplementedError(
                "dense MLA path requires a flashmla_backend; pass one to "
                "DeepseekV4AttnBackend(..., flashmla_backend=...)"
            )
        return self._flashmla(
            q=q,
            kv=kv,
            attn_sink=attn_sink,
            metadata=self.metadata,
            layer_idx=layer_config.layer_idx,
            **kwargs,
        )

    def _forward_c4_sparse(
        self,
        layer_config: DSV4LayerConfig,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: Optional[torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        if self._fused_indexer is None:
            from batchgen_kernels.attention.dsa.fused_indexer_score import (
                fused_score_and_topk,
            )

            self._fused_indexer = fused_score_and_topk

        meta = self.metadata
        if meta.c4_out_loc is None or meta.c4_topk_lengths_clamp1 is None:
            raise RuntimeError(
                "c4 sparse path requires meta.c4_out_loc and "
                "meta.c4_topk_lengths_clamp1 to be populated"
            )
        if self._flashmla is None:
            raise NotImplementedError(
                "c4 sparse decode requires a flashmla_backend for the sparse "
                "FlashMLA call after top-512 selection"
            )

        head_gates = kwargs.pop("head_gates", None)
        if head_gates is None:
            raise ValueError(
                "c4 sparse path requires head_gates: pass kwargs['head_gates']"
            )

        q_attn = kwargs.pop("q_attn", q)
        current_kv = kwargs.pop("current_kv", kv)

        top_k_indices = self._fused_indexer(
            q=q,
            cached_k=kv,
            head_gates=head_gates,
            cache_seqlens=meta.c4_topk_lengths_clamp1,
            topk=meta.c4_sparse_topk,
        )

        return self._flashmla(
            q=q_attn,
            kv=current_kv,
            attn_sink=attn_sink,
            metadata=meta,
            layer_idx=layer_config.layer_idx,
            sparse_indices=top_k_indices,
            **kwargs,
        )

    def _forward_c128_compress(
        self,
        layer_config: DSV4LayerConfig,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: Optional[torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        if self._compressor is None:
            from batchgen_kernels.attention.v4_compressor import (
                DeepSeekV4Compressor,
            )

            self._compressor = DeepSeekV4Compressor

        meta = self.metadata
        if (
            meta.c128_page_indices is None
            or meta.c128_topk_lengths_clamp1 is None
            or meta.c128_out_loc is None
        ):
            raise RuntimeError(
                "c128 compressed path requires meta.c128_page_indices, "
                "meta.c128_topk_lengths_clamp1, and meta.c128_out_loc to be populated"
            )
        if self._flashmla is None:
            raise NotImplementedError(
                "c128 compressed decode requires a flashmla_backend for the "
                "compressed FlashMLA call after HCA compression"
            )

        return self._flashmla(
            q=q,
            kv=kv,
            attn_sink=attn_sink,
            metadata=meta,
            layer_idx=layer_config.layer_idx,
            compressed_page_indices=meta.c128_page_indices,
            compressed_lengths=meta.c128_topk_lengths_clamp1,
            **kwargs,
        )


def build_layer_configs_from_compress_ratios(
    compress_ratios: list[int],
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    swa_window: int = SWA_WINDOW,
) -> list[DSV4LayerConfig]:
    return [
        DSV4LayerConfig(
            layer_idx=i,
            compress_ratio=r,
            n_heads=n_heads,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            swa_window=swa_window,
        )
        for i, r in enumerate(compress_ratios)
    ]


__all__ = [
    "C4_TOPK",
    "PAGE_INDEX_ALIGNED_SIZE",
    "SWA_WINDOW",
    "DSV4AttnMetadata",
    "DSV4LayerConfig",
    "DeepseekV4AttnBackend",
    "V4AttnPath",
    "build_layer_configs_from_compress_ratios",
]
