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


@dataclass(frozen=True)
class ChunkBatch:
    """The tensors one prepacked forward needs; rows follow ``items``."""

    input_ids: torch.Tensor  # int64 [T]
    position_ids: torch.Tensor  # int64 [T], absolute positions of the item
    cu_seqlens_q: torch.Tensor  # int32 [n + 1]
    cache_seqlens: torch.Tensor  # int32 [n], prefix plus own tokens
    page_table: torch.Tensor  # int32 [n, W], ancestors then own pages, 0 pad
    slots: torch.Tensor  # int64 [T], where each query token's KV is written
    tail_rows: torch.Tensor  # int64, flat index of each tail's last token
    tail_wave_indices: tuple[int, ...]  # wave index per entry of tail_rows
    items: tuple["PlanItem", ...]


class PoolChunkExecutor:
    """Replay a plan chunk by chunk against one prefill prefix pool."""

    def __init__(
        self,
        plan: "WavePrefixPlan",
        prompts: Sequence[Sequence[int]],
        pool: "PrefillPrefixPool",
    ) -> None:
        if plan.block_tokens != pool.page_tokens:
            raise ValueError(
                f"plan block_tokens {plan.block_tokens} != pool page_tokens "
                f"{pool.page_tokens}"
            )
        self.plan = plan
        self.prompts = prompts
        self.pool = pool
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
        table = [row + [0] * (width - len(row)) for row in rows]
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
