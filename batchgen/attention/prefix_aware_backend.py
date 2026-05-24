"""Prefix-aware attention backend adapters.

The adapters in this module provide a small explicit interface for prefill
attention where query length and KV length can differ because cached prefix KV
is prepended to freshly computed suffix KV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import torch


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
    """GQA backend adapter for varlen prefill and paged extend prefill."""

    prefix_kv_builder: object
    num_kv_heads: int
    head_dim: int
    sinks: Optional[torch.Tensor] = None
    softmax_scale: Optional[float] = None
    sliding_window: Optional[int] = None
    attention_fn: Optional[Callable[..., tuple[torch.Tensor, object]]] = None

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

        cu_q = metadata.cu_seqlens.to(query.device)
        materialization = (
            getattr(kv_cache_metadata, "prefill_prefix_materialization", None)
            if kv_cache_metadata is not None
            else None
        )
        if metadata.full_hit_mode and materialization is None:
            raise RuntimeError(
                "GQA full-hit prefix reuse requires GPU paged materialization"
            )
        if metadata.prefix_reuse_mode and materialization is None:
            raise RuntimeError(
                "GQA partial-hit prefix reuse requires GPU paged materialization"
            )

        if metadata.full_hit_mode:
            return self._forward_paged_full_hit_prefill(
                query=query,
                metadata=metadata,
                materialization=materialization,
            )
        if metadata.prefix_reuse_mode:
            return self._forward_paged_extend_prefill(
                query=query,
                key=key,
                value=value,
                metadata=metadata,
                materialization=materialization,
            )

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
        return attn_output

    def _forward_paged_extend_prefill(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata,
        materialization,
    ) -> torch.Tensor:
        """Run prefix-hit suffix prefill over materialized GPU paged KV."""

        from batchgen.attention.gqa import gqa_extend_fa

        layer_idx = int(self.prefix_kv_builder.reader.layer_idx)
        materialization.wait_for_layer(layer_idx)
        materialization.manager.append_layer_prefill_suffix_tokens(
            k_tensor=key,
            v_tensor=value,
            append_plan=materialization.append_plan,
            layer_idx=layer_idx,
        )
        k_cache, v_cache, page_table = (
            materialization.manager.get_layer_kv_with_page_table(layer_idx)
        )
        if v_cache is None:
            raise RuntimeError("GQA paged prefix prefill requires V cache")

        cu_k = torch.nn.functional.pad(
            torch.cumsum(
                materialization.append_plan.cache_seqlens,
                dim=0,
                dtype=torch.int32,
            ),
            (1, 0),
        )
        attn_output, _ = gqa_extend_fa(
            q=query,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=materialization.append_plan.cache_seqlens,
            page_table=page_table,
            cu_seqlens_q=metadata.cu_seqlens.to(
                device=query.device,
                dtype=torch.int32,
            ),
            cu_seqlens_k=cu_k,
            max_seqlen_q=int(metadata.max_seqlen),
            sinks=self.sinks,
            softmax_scale=self.softmax_scale,
            sliding_window=self.sliding_window,
        )
        return attn_output

    def _forward_paged_full_hit_prefill(
        self,
        *,
        query: torch.Tensor,
        metadata,
        materialization,
    ) -> torch.Tensor:
        """Run exact full-hit prefill over materialized GPU paged KV."""

        from batchgen.attention.gqa import gqa_decode_fa

        layer_idx = int(self.prefix_kv_builder.reader.layer_idx)
        materialization.wait_for_layer(layer_idx)
        k_cache, v_cache, page_table = (
            materialization.manager.get_layer_kv_with_page_table(layer_idx)
        )
        if v_cache is None:
            raise RuntimeError("GQA paged full-hit prefill requires V cache")

        slot_indices = materialization.append_plan.slot_values
        if isinstance(slot_indices, torch.Tensor):
            slot_indices_tensor = slot_indices.to(
                device=page_table.device,
                dtype=torch.long,
            )
        else:
            slot_indices_tensor = torch.tensor(
                [int(slot_idx) for slot_idx in slot_indices],
                dtype=torch.long,
                device=page_table.device,
            )
        block_table = page_table.index_select(0, slot_indices_tensor)
        cache_seqlens = torch.tensor(
            [int(seq_len) for seq_len in metadata.full_seq_lengths],
            dtype=torch.int32,
            device=query.device,
        )

        squeeze_query_dim = False
        if query.ndim == 3:
            decode_query = query.unsqueeze(1)
            squeeze_query_dim = True
        elif query.ndim == 4 and query.shape[1] == 1:
            decode_query = query
        else:
            raise RuntimeError(
                "GQA full-hit prefix reuse expects query shape "
                f"[batch, heads, dim] or [batch, 1, heads, dim], got "
                f"{tuple(query.shape)}"
            )

        attn_output, _ = gqa_decode_fa(
            q=decode_query,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens,
            block_table=block_table,
            sinks=self.sinks,
            softmax_scale=self.softmax_scale,
            sliding_window=self.sliding_window,
        )
        if squeeze_query_dim:
            return attn_output.squeeze(1)
        return attn_output


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
        from batchgen.models.wrappers.prefix_mla_replay import (
            MlaReplaySpec,
            run_projected_mla_prefix_attention,
        )

        materialization = (
            getattr(kv_cache_metadata, "prefill_prefix_materialization", None)
            if kv_cache_metadata is not None
            else None
        )

        spec = MlaReplaySpec(
            kv_dim=int(self.kv_dim),
            num_heads=int(self.num_heads),
            kv_lora_rank=int(self.kv_lora_rank),
            softmax_scale=float(self.softmax_scale),
        )
        attn_out = run_projected_mla_prefix_attention(
            prefix_kv_builder=self.prefix_kv_builder,
            query_states=query,
            offload_kv=key,
            metadata=metadata,
            spec=spec,
            page_size=int(self.page_size),
            attention_fn=self.attention_fn,
            prefill_prefix_materialization=materialization,
        )
        if self.output_projection is None:
            return attn_out
        return self.output_projection(attn_out)
