"""Turn the chunks of a wave prefix plan into prepacked forward batches.

One chunk becomes one prepacked forward: every item contributes its own query
tokens, and attention reads the item's pooled ancestors as the already
computed prefix ``[0, token_start)``. The executor owns the pool pages: a
pooled segment keeps its pages until the plan releases it, a tail's pages are
scratch that the chunk gives back.

Pure bookkeeping: index math runs on CPU lists, one tensor per field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch

if TYPE_CHECKING:
    from batchgen.prefix_reuse.pool import PrefillPrefixPool
    from batchgen.prefix_reuse.wave_plan import PlanChunk, PlanItem, WavePrefixPlan


# Batch-level ``batchgen_debug`` flag, read by the worker when it builds the
# executor. DIAGNOSTIC ONLY: see ``PoolChunkExecutor.__init__``.
PAD_POISON_DEBUG_FLAG = "prefix_pool_pad_poison"


@dataclass(frozen=True)
class ChunkBatch:
    """The tensors one prepacked forward needs; rows follow ``items``."""

    input_ids: torch.Tensor  # int64 [T]
    position_ids: torch.Tensor  # int64 [T], absolute positions of the item
    cu_seqlens_q: torch.Tensor  # int32 [n + 1]
    cache_seqlens: torch.Tensor  # int32 [n], prefix plus own tokens
    # int32 [n, W], ancestors then own pages; padded to W with 0, or with the
    # row's own last page under the prefix_pool_pad_poison probe.
    page_table: torch.Tensor
    slots: torch.Tensor  # int64 [T], where each query token's KV is written
    tail_rows: torch.Tensor  # int64, flat index of each tail's last token
    tail_wave_indices: tuple[int, ...]  # wave index per entry of tail_rows
    items: tuple["PlanItem", ...]
    row_bounds: tuple[int, ...]  # cu_seqlens_q on the host
    max_seqlen_q: int
    max_cache_seqlen: int


class PoolChunkExecutor:
    """Replay a plan chunk by chunk against one prefill prefix pool."""

    def __init__(
        self,
        plan: "WavePrefixPlan",
        prompts: Sequence[Sequence[int]],
        pool: "PrefillPrefixPool",
        *,
        pad_poison: bool = False,
    ) -> None:
        """``pad_poison`` is the batch-level ``prefix_pool_pad_poison`` probe.

        A chunk's page table is one rectangle, so a row shorter than the
        widest one is padded. The pad value is 0, which is a REAL pool page -
        normally the one holding the start of the root segment. A kernel that
        honours ``cache_seqlens`` never reads those columns, so the value is
        free; one that derives its page count from the table width instead
        reads the root segment's first page as if it were this row's context.

        With the flag on, each row is padded with ITS OWN last valid page id
        instead. That is equally unread under a correct kernel, so output
        parity between off and on exonerates the padding; a change in output
        proves padding is being read, and the widest-padded rows are the
        shortest members - which is where the live corruption appeared.

        Off by default: the packed table is then byte-identical to before.
        """

        if plan.block_tokens != pool.page_tokens:
            raise ValueError(
                f"plan block_tokens {plan.block_tokens} != pool page_tokens "
                f"{pool.page_tokens}"
            )
        self.plan = plan
        self.prompts = prompts
        self.pool = pool
        self.pad_poison = bool(pad_poison)
        self.segment_pages: dict[int, list[int]] = {}
        self._tail_pages: list[int] = []

    def build(self, chunk: "PlanChunk") -> ChunkBatch:
        page_tokens = self.pool.page_tokens
        input_ids: list[int] = []
        position_ids: list[int] = []
        cu_seqlens: list[int] = [0]
        cache_seqlens: list[int] = []
        rows: list[list[int]] = []
        slots: list[int] = []
        tail_rows: list[int] = []
        tail_waves: list[int] = []
        for item in chunk.items:
            row: list[int] = []
            for seg in item.prefix_segments:
                pages = self.segment_pages.get(seg)
                if pages is None:
                    raise RuntimeError(f"segment {seg} is not resident in the pool")
                row.extend(pages)
            own = self.pool.alloc(-(-item.tokens // page_tokens))
            if item.kind == "segment":
                self.segment_pages[item.ref] = own
                tokens = self.prompts[self.plan.segments[item.ref].members[0]]
            else:
                self._tail_pages.extend(own)
                tokens = self.prompts[item.ref]
                tail_rows.append(len(input_ids) + item.tokens - 1)
                tail_waves.append(item.ref)
            row.extend(own)
            rows.append(row)
            input_ids.extend(tokens[item.token_start:item.token_end])
            position_ids.extend(range(item.token_start, item.token_end))
            cu_seqlens.append(len(input_ids))
            cache_seqlens.append(item.token_end)
            slots.extend(self.pool.slot_list(own, 0, item.tokens))

        width = max(len(row) for row in rows)
        # Every row owns at least one page, so row[-1] is always a valid id.
        table = [
            row + [row[-1] if self.pad_poison else 0] * (width - len(row))
            for row in rows
        ]
        device = self.pool.device
        return ChunkBatch(
            input_ids=torch.tensor(input_ids, dtype=torch.long, device=device),
            position_ids=torch.tensor(position_ids, dtype=torch.long, device=device),
            cu_seqlens_q=torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
            cache_seqlens=torch.tensor(cache_seqlens, dtype=torch.int32, device=device),
            page_table=torch.tensor(table, dtype=torch.int32, device=device),
            slots=torch.tensor(slots, dtype=torch.long, device=device),
            tail_rows=torch.tensor(tail_rows, dtype=torch.long, device=device),
            tail_wave_indices=tuple(tail_waves),
            items=chunk.items,
            row_bounds=tuple(cu_seqlens),
            max_seqlen_q=max(item.tokens for item in chunk.items),
            max_cache_seqlen=max(cache_seqlens),
        )

    def finish(self, chunk: "PlanChunk") -> None:
        """Give back this chunk's scratch and the segments it completed."""

        self.pool.free(self._tail_pages)
        self._tail_pages = []
        for seg in chunk.release_after:
            self.pool.free(self.segment_pages.pop(seg))
        live = sum(len(pages) for pages in self.segment_pages.values())
        if self.pool.free_pages != self.pool.num_pages - live:
            raise RuntimeError(
                f"prefix pool accounting: free={self.pool.free_pages} "
                f"total={self.pool.num_pages} live_segment_pages={live}"
            )

    def close(self) -> None:
        if self.segment_pages or self.pool.free_pages != self.pool.num_pages:
            raise RuntimeError(
                f"prefix pool not empty after the wave: live segments "
                f"{sorted(self.segment_pages)}, free={self.pool.free_pages} "
                f"total={self.pool.num_pages}"
            )


def pool_prefill_attention(
    *,
    pool: "PrefillPrefixPool",
    layer_idx: int,
    batch: ChunkBatch,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks,
    softmax_scale: float,
    sliding_window,
):
    """Write one layer of a chunk's K/V into the pool, then attend.

    Every row reads its pooled ancestors plus its own tokens through its page
    table, so the write must cover the whole chunk before attention runs.

    K and V are indexed with their own head dim: a GQA model keeps one dim for
    both, an MLA model attending on the expanded heads keeps a wider K than V
    (192 against 128 per head on Kimi-K3).
    """

    from batchgen.attention.gqa import gqa_extend_fa

    k_cache, v_cache = pool.k[layer_idx], pool.v[layer_idx]
    k_cache.view(-1, *k_cache.shape[-2:]).index_copy_(0, batch.slots, key)
    v_cache.view(-1, *v_cache.shape[-2:]).index_copy_(0, batch.slots, value)
    return gqa_extend_fa(
        q=query,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=batch.cache_seqlens,
        page_table=batch.page_table,
        cu_seqlens_q=batch.cu_seqlens_q,
        max_seqlen_q=batch.max_seqlen_q,
        sinks=sinks,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
    )
