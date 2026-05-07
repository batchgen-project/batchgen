"""Prefix-aware attention backend adapters.

The adapters in this module provide a small explicit interface for prefill
attention where query length and KV length can differ because cached prefix KV
is prepended to freshly computed suffix KV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import torch

from batchgen.attention.prefix_gpu_extend import (
    append_suffix_to_gpu_kv,
    current_kv_cache_metadata,
    gpu_page_table_attention_enabled,
    gqa_prefill_with_gpu_paged_kv,
    mla_prefill_with_gpu_paged_kv,
)


class PrefixAwareAttentionBackend(Protocol):
    """Common protocol for prefix-aware prefill attention backends."""

    def forward_prefill(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor],
        metadata,
        kv_cache_metadata=None,
    ) -> torch.Tensor:
        """Run prefill attention for a possibly prefix-reused batch."""


@dataclass(frozen=True)
class GqaPrefixAwareAttentionBackend:
    """GQA backend adapter using existing varlen FlashAttention implementation."""

    prefix_kv_builder: object
    num_kv_heads: int
    head_dim: int
    sinks: Optional[torch.Tensor] = None
    softmax_scale: Optional[float] = None
    sliding_window: Optional[int] = None
    attention_fn: Optional[Callable[..., tuple[torch.Tensor, object]]] = None
    paged_attention_fn: Optional[Callable[..., tuple[torch.Tensor, object]]] = None
    layer_idx: Optional[int] = None
    enable_gpu_suffix_append: bool = False

    def forward_prefill(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor],
        metadata,
        kv_cache_metadata=None,
    ) -> torch.Tensor:
        if value is None:
            raise RuntimeError("GQA prefix-aware prefill requires value tensor")

        from batchgen.models.wrappers.prefix_cache import (
            ensure_prefix_cache_prepack_metadata,
        )

        metadata = ensure_prefix_cache_prepack_metadata(metadata)
        if gpu_page_table_attention_enabled() and metadata.prefix_reuse_mode:
            return gqa_prefill_with_gpu_paged_kv(
                query=query,
                key=key,
                value=value,
                metadata=metadata,
                kv_cache_metadata=kv_cache_metadata,
                layer_idx=self.layer_idx,
                paged_attention_fn=self.paged_attention_fn,
                sinks=self.sinks,
                softmax_scale=self.softmax_scale,
                sliding_window=self.sliding_window,
            )

        cu_q = metadata.cu_seqlens.to(query.device)
        if metadata.full_hit_mode:
            key_for_attn, value_for_attn, cu_k, max_seqlen_k = (
                self.prefix_kv_builder.build_gqa_full_hit_kv(
                    metadata=metadata,
                    num_heads=int(self.num_kv_heads),
                    head_dim=int(self.head_dim),
                    dtype=key.dtype,
                    device=key.device,
                )
            )
        elif metadata.prefix_reuse_mode:
            key_for_attn, value_for_attn, cu_k, max_seqlen_k = (
                self.prefix_kv_builder.build_gqa_prefix_kv(
                    key=key,
                    value=value,
                    metadata=metadata,
                    num_heads=int(self.num_kv_heads),
                    head_dim=int(self.head_dim),
                )
            )
        else:
            key_for_attn = key
            value_for_attn = value
            cu_k = cu_q
            max_seqlen_k = metadata.max_seqlen

        attention_fn = self.attention_fn
        if attention_fn is None:
            from batchgen.attention.gqa import gqa_prefill_fa

            attention_fn = gqa_prefill_fa
        attn_output, _ = attention_fn(
            q=query,
            k=key_for_attn,
            v=value_for_attn,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=metadata.max_seqlen,
            max_seqlen_k=max_seqlen_k,
            sinks=self.sinks,
            softmax_scale=self.softmax_scale,
            sliding_window=self.sliding_window,
        )
        self._maybe_append_suffix_to_gpu_kv(
            key=key,
            value=value,
            metadata=metadata,
            kv_cache_metadata=kv_cache_metadata,
        )
        return attn_output

    def _maybe_append_suffix_to_gpu_kv(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata,
        kv_cache_metadata,
    ) -> None:
        import os

        enabled = self.enable_gpu_suffix_append or (
            os.environ.get("BATCHGEN_PREFIX_REUSE_GPU_EXTEND_WRITES", "0") == "1"
        )
        if not enabled:
            return
        if metadata.full_hit_mode:
            return
        if self.layer_idx is None:
            raise RuntimeError(
                "GQA GPU suffix append requires layer_idx on the backend"
            )
        if kv_cache_metadata is None:
            kv_cache_metadata = current_kv_cache_metadata()
        append_suffix_to_gpu_kv(
            kv_cache_metadata=kv_cache_metadata,
            k_tensor=key,
            v_tensor=value,
            layer_idx=int(self.layer_idx),
            metadata=metadata,
            manager_attr="gpu_paged_kv_manager",
            context="GQA GPU suffix append",
        )


@dataclass(frozen=True)
class MlaProjectedPrefixAwareAttentionBackend:
    """MLA backend adapter for already projected query and compressed KV."""

    prefix_kv_builder: object
    page_size: int
    kv_dim: int
    num_heads: int
    kv_lora_rank: int
    softmax_scale: float
    output_projection: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    attention_fn: Optional[Callable[..., torch.Tensor]] = None
    layer_idx: Optional[int] = None
    enable_gpu_suffix_append: bool = False

    def forward_prefill(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor],
        metadata,
        kv_cache_metadata=None,
    ) -> torch.Tensor:
        del value
        from batchgen.models.wrappers.prefix_cache import (
            ensure_prefix_cache_prepack_metadata,
        )
        from batchgen.models.wrappers.prefix_mla_replay import (
            MlaReplaySpec,
            block_mla_kv_by_sequence,
            run_flash_mla_prefix_attention,
        )

        metadata = ensure_prefix_cache_prepack_metadata(metadata)
        if gpu_page_table_attention_enabled() and metadata.prefix_reuse_mode:
            attn_out = mla_prefill_with_gpu_paged_kv(
                query=query,
                key=key,
                metadata=metadata,
                kv_cache_metadata=kv_cache_metadata,
                layer_idx=self.layer_idx,
                kv_dim=int(self.kv_dim),
                num_heads=int(self.num_heads),
                kv_lora_rank=int(self.kv_lora_rank),
                softmax_scale=float(self.softmax_scale),
                attention_fn=self.attention_fn,
            )
            if self.output_projection is None:
                return attn_out
            return self.output_projection(attn_out)

        if metadata.full_hit_mode:
            compressed_kv, cu_k, _ = self.prefix_kv_builder.build_mla_full_hit_kv(
                metadata=metadata,
                kv_dim=int(self.kv_dim),
                dtype=query.dtype,
                device=query.device,
            )
            query_len = 1
        elif metadata.prefix_reuse_mode:
            compressed_kv, cu_k, _ = self.prefix_kv_builder.build_mla_prefix_kv(
                key=key,
                metadata=metadata,
                kv_dim=int(self.kv_dim),
            )
            query_len = int(metadata.max_seqlen)
        else:
            compressed_kv = key
            if compressed_kv.dim() == 2:
                compressed_kv = compressed_kv.unsqueeze(1)
            cu_k = metadata.cu_seqlens.to(compressed_kv.device)
            query_len = int(metadata.max_seqlen)

        blocked_k, block_table, cache_seqlens = block_mla_kv_by_sequence(
            compressed_kv=compressed_kv,
            cu_k=cu_k,
            page_size=int(self.page_size),
        )
        spec = MlaReplaySpec(
            kv_dim=int(self.kv_dim),
            num_heads=int(self.num_heads),
            kv_lora_rank=int(self.kv_lora_rank),
            softmax_scale=float(self.softmax_scale),
        )
        attention_fn = self.attention_fn or run_flash_mla_prefix_attention
        attn_out = attention_fn(
            query_states=query,
            blocked_k=blocked_k,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            query_len=query_len,
            spec=spec,
        )
        self._maybe_append_suffix_to_gpu_kv(
            key=key,
            metadata=metadata,
            kv_cache_metadata=kv_cache_metadata,
        )
        if self.output_projection is None:
            return attn_out
        return self.output_projection(attn_out)

    def _maybe_append_suffix_to_gpu_kv(
        self,
        *,
        key: torch.Tensor,
        metadata,
        kv_cache_metadata,
    ) -> None:
        import os

        enabled = self.enable_gpu_suffix_append or (
            os.environ.get("BATCHGEN_PREFIX_REUSE_GPU_EXTEND_WRITES", "0") == "1"
        )
        if not enabled:
            return
        if metadata.full_hit_mode:
            return
        if self.layer_idx is None:
            raise RuntimeError(
                "MLA GPU suffix append requires layer_idx on the backend"
            )
        if kv_cache_metadata is None:
            kv_cache_metadata = current_kv_cache_metadata()
        append_suffix_to_gpu_kv(
            kv_cache_metadata=kv_cache_metadata,
            k_tensor=key,
            v_tensor=None,
            layer_idx=int(self.layer_idx),
            metadata=metadata,
            manager_attr="gpu_paged_kv_manager",
            context="MLA GPU suffix append",
        )
