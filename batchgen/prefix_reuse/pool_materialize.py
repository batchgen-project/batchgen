"""Load a committed prefix's Host KV into the in-wave prefix pool.

A trajectory re-entry hit's prefix is already KV: the wave that computed it
published and retained its Host pages. This module brings those pages back to
the device in bounded slabs; the MODEL then expands each slab into the pool's
own layout, through its own projection, because what the Host holds (one
latent per token per layer) is not what the pool holds (per-head K and V).

Split of duties, deliberately:

* here -- which Host pages, in what order, into how large a staging buffer,
  and the one wait that makes the bytes visible to the expansion;
* the model -- the projection and the write into the pool slots, so the loaded
  prefix is bit-for-bit what computing it would have written, and nothing
  re-implements a GEMM the forward already owns.

The H2D is one async task per slab over ALL layers at once (the pointer
tensors the Host view takes are indexed by physical layer), which is the
right shape for a prefix: its volume is a page count, not a layer count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import torch


# Transient device staging for the H2D, sized in Host pages across every
# layer (a K3 page is ~1.9 MiB over its 27 MLA layers, so this is a handful of
# pages). It is NOT part of the pool's measured page budget, which is why it
# is a small constant rather than a fraction of free memory: a load is
# bandwidth-bound, and a deeper buffer buys nothing once the copy is in
# flight.
DEFAULT_SLAB_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class LatentSlab:
    """One resident slab of a segment, on the device and ready to expand.

    ``latent`` is ``[layers, tokens, latent_dim]`` -- a VIEW of the loader's
    staging buffer, valid until the next slab is loaded. ``token_start`` is
    the slab's offset inside the segment, so the caller slices the segment's
    pool slots with it.
    """

    token_start: int
    tokens: int
    latent: torch.Tensor


class MaterializedSegmentLoader:
    """Reusable H2D staging for the materialized segments of one wave.

    One buffer for the whole wave: segments are loaded one slab at a time and
    expanded before the next load, so the peak cost is one slab -- bounded by
    ``max_slab_bytes`` -- and not one prefix. A 10k-token frontier would
    otherwise stage hundreds of MB next to a pool whose own budget was
    measured from free memory.
    """

    def __init__(
        self,
        *,
        worker_view,
        num_layers: int,
        page_tokens: int,
        latent_dim: int,
        dtype: torch.dtype,
        device,
        max_slab_bytes: int,
        torch_module=None,
    ) -> None:
        self._torch = torch_module if torch_module is not None else torch
        self._view = worker_view
        self._num_layers = int(num_layers)
        self._page_tokens = int(page_tokens)
        self._latent_dim = int(latent_dim)
        if self._num_layers <= 0 or self._page_tokens <= 0 or self._latent_dim <= 0:
            raise ValueError(
                "a materialized segment loader needs a positive layer count, "
                f"page size and latent width, got layers={num_layers}, "
                f"page_tokens={page_tokens}, latent_dim={latent_dim}"
            )
        element = self._torch.empty((), dtype=dtype).element_size()
        self._page_bytes = self._page_tokens * self._latent_dim * element
        slab_pages = int(max_slab_bytes) // (self._page_bytes * self._num_layers)
        if slab_pages < 1:
            raise ValueError(
                f"a staging budget of {int(max_slab_bytes)} B holds no page of "
                f"{self._page_bytes * self._num_layers} B across "
                f"{self._num_layers} layers"
            )
        self._slab_pages = slab_pages
        self._staging = self._torch.empty(
            (self._num_layers, slab_pages * self._page_tokens, self._latent_dim),
            dtype=dtype,
            device=device,
        )
        self._layer_stride = slab_pages * self._page_bytes

    @property
    def slab_pages(self) -> int:
        return self._slab_pages

    @property
    def staging_bytes(self) -> int:
        return self._staging.numel() * self._staging.element_size()

    def slabs(
        self, *, host_page_ids: Sequence[int], tokens: int
    ) -> Iterator[LatentSlab]:
        """Load the segment slab by slab; yields each slab once resident.

        The caller must finish with a slab before taking the next one: the
        staging buffer is reused, which is the whole point of slabbing.
        """

        torch_module = self._torch
        pages = [int(page) for page in host_page_ids]
        expected = int(tokens) // self._page_tokens
        if int(tokens) % self._page_tokens or expected != len(pages):
            raise ValueError(
                f"a materialized segment of {int(tokens)} tokens is "
                f"{expected} page(s) of {self._page_tokens}, but "
                f"{len(pages)} Host page(s) were named"
            )
        base = self._staging.data_ptr()
        for first in range(0, len(pages), self._slab_pages):
            window = pages[first:first + self._slab_pages]
            count = len(window)
            host_page_tensor = torch_module.tensor(
                [window], dtype=torch_module.int64
            )
            active = torch_module.tensor([count], dtype=torch_module.int64)
            pointers = torch_module.tensor(
                [
                    [[
                        base + layer * self._layer_stride + page * self._page_bytes
                        for page in range(count)
                    ]]
                    for layer in range(self._num_layers)
                ],
                dtype=torch_module.int64,
            )
            task = self._view.async_load_prefix_pages_to_device(
                host_page_ids=host_page_tensor,
                active_page_counts=active,
                k_device_ptrs=pointers,
                v_device_ptrs=None,
            )
            # The expansion reads these bytes on the compute stream in the
            # very next statement, so the load is waited on here rather than
            # tracked: a materialization that raced its own reader would
            # expand pinned memory nobody has written.
            task.wait()
            slab_tokens = count * self._page_tokens
            yield LatentSlab(
                token_start=first * self._page_tokens,
                tokens=slab_tokens,
                latent=self._staging[:, :slab_tokens, :],
            )

    def release(self) -> None:
        self._staging = None
