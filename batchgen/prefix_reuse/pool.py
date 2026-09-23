"""The paged KV pool one DP rank uses for one prefill phase.

The wave plan decides which shared segments are resident when; this pool owns
the pages that hold them. It exists only while a prefill phase runs, so the
always-on identity is that every page is free again before it is released.

The page layout mirrors the GPU paged KV cache
``(num_layers, num_pages, page_tokens, num_kv_heads, head_dim)``, so a slot
index is ``page * page_tokens + offset`` exactly as in the main cache.
"""

from __future__ import annotations

import bisect
from typing import Sequence

import torch


def pool_pages_from_budget(
    free_bytes: int, workspace_bytes: int, page_bytes: int
) -> int:
    """How many pool pages fit in the free memory the workspace leaves over."""

    if page_bytes <= 0:
        raise ValueError("page_bytes must be positive")
    if free_bytes < workspace_bytes:
        raise ValueError("free_bytes must cover workspace_bytes")
    return (free_bytes - workspace_bytes) // page_bytes


def scratch_pages_bound(
    chunk_tokens: int, longest_prompt: int, num_prompts: int, page_tokens: int
) -> int:
    """Exact upper bound on the tail pages alive inside one chunk.

    A chunk packs at most ``chunk_tokens`` tokens unless a single item is
    larger, which caps it at the longest prompt; a chunk holds at most one tail
    per prompt and each tail adds at most one partially filled page.
    """

    tokens = max(chunk_tokens, longest_prompt)
    return -(-tokens // page_tokens) + num_prompts


class PrefillPrefixPool:
    """Pages of KV for the pooled segments and the in-flight tails."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_pages: int,
        page_tokens: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        self.page_tokens = page_tokens
        self.device = torch.device(device)
        shape = (num_layers, num_pages, page_tokens, num_kv_heads, head_dim)
        self.k = torch.empty(shape, dtype=dtype, device=self.device)
        self.v = torch.empty(shape, dtype=dtype, device=self.device)
        self._num_pages = num_pages
        self._free = list(range(num_pages))  # kept ascending
        self._page_bytes = (
            2 * num_layers * page_tokens * num_kv_heads * head_dim
            * self.k.element_size()
        )

    @property
    def page_bytes(self) -> int:
        """Bytes of K and V for one page across all layers."""

        return self._page_bytes

    @property
    def num_pages(self) -> int:
        return self._num_pages

    @property
    def free_pages(self) -> int:
        return len(self._free)

    def alloc(self, n: int) -> list[int]:
        """Take the ``n`` lowest free pages, so allocation is deterministic."""

        if n > len(self._free):
            raise RuntimeError(
                f"prefix pool exhausted: need {n} pages, have {len(self._free)}"
            )
        pages = self._free[:n]
        del self._free[:n]
        return pages

    def free(self, pages: Sequence[int]) -> None:
        for page in pages:
            if not 0 <= page < self._num_pages:
                raise RuntimeError(f"page {page} is not a pool page")
            at = bisect.bisect_left(self._free, page)
            if at < len(self._free) and self._free[at] == page:
                raise RuntimeError(f"page {page} is already free")
            self._free.insert(at, page)

    def slot_list(
        self, pages: Sequence[int], start_offset: int, n_tokens: int
    ) -> list[int]:
        """Flat slot indices for ``n_tokens`` tokens from ``start_offset``.

        Token offsets run over the concatenation of ``pages``.
        """

        if start_offset + n_tokens > len(pages) * self.page_tokens:
            raise ValueError(
                f"{n_tokens} tokens at offset {start_offset} do not fit in "
                f"{len(pages)} pages of {self.page_tokens}"
            )
        return [
            pages[offset // self.page_tokens] * self.page_tokens
            + offset % self.page_tokens
            for offset in range(start_offset, start_offset + n_tokens)
        ]

    def slots(
        self, pages: Sequence[int], start_offset: int, n_tokens: int
    ) -> torch.Tensor:
        return torch.tensor(
            self.slot_list(pages, start_offset, n_tokens),
            dtype=torch.long, device=self.device,
        )

    def release(self) -> None:
        """Drop the tensors; nothing may still hold a page."""

        if len(self._free) != self._num_pages:
            raise RuntimeError(
                f"{self._num_pages - len(self._free)} pool pages still allocated"
            )
        self.k = None
        self.v = None
