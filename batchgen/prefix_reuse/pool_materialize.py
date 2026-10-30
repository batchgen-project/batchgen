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


# --------------------------------------------------------------------------- #
#  Dual-region (DSA) load: the MLA latent AND the indexer keys of one prefix
# --------------------------------------------------------------------------- #
#
# A DSA model (GLM-5.x) keeps a token's KV in TWO Host regions: the MLA latent
# in the primary region (prefix-cache group 0) and the DSA indexer keys in the
# auxiliary one (group 1). Its pool holds exactly those two things -- the
# latent of every layer in ``pool.k`` and the indexer keys of its top-k
# layers in ``pool.indexer`` -- in the bytes the forward's own offload wrote to
# Host. A load is therefore a COPY of both regions into the pool, never a
# projection, and it must bring both: a latent loaded without its indexer
# keys leaves the indexer scoring whatever the pool slots held before, which
# is wrong top-k with no error.

PRIMARY_GROUP = 0
AUX_GROUP = 1


class DualRegionPageMismatch(RuntimeError):
    """The two Host regions do not name the same pages for one prefix.

    ``DualHostKVCoordinator`` mirrors every allocation into both regions, so
    a prefix's pages are the same ids in the same order in each. A load that
    disagrees is refused by name: reading the indexer keys of other tokens is
    wrong output, not a recoverable miss.
    """


def aligned_region_pages(
    pages_by_group, *, groups: Sequence[int], page_tokens: int,
    tokens: int | None = None,
) -> tuple[int, ...]:
    """The one page list every Host region names for a prefix, or raise.

    Requires exactly ``groups``, identical ids in identical order in every
    group and, given ``tokens``, a page count that covers them exactly.
    """

    named = sorted(int(group) for group in pages_by_group)
    wanted = sorted(int(group) for group in groups)
    if named != wanted:
        raise DualRegionPageMismatch(
            f"a dual-region prefix names Host KV groups {named}; the regions "
            f"are {wanted}"
        )
    by_group = {
        int(group): tuple(int(page) for page in pages)
        for group, pages in pages_by_group.items()
    }
    reference = by_group[wanted[0]]
    for group in wanted[1:]:
        other = by_group[group]
        if other == reference:
            continue
        if len(other) != len(reference):
            detail = f"{len(reference)} page(s) against {len(other)}"
        else:
            at = next(
                index for index, (a, b) in enumerate(zip(reference, other))
                if a != b
            )
            detail = (
                f"page {at} is {reference[at]} in group {wanted[0]} and "
                f"{other[at]} in group {group}"
            )
        raise DualRegionPageMismatch(
            f"Host KV groups {wanted[0]} and {group} disagree on the pages of "
            f"one prefix ({detail}); both regions must hold the same tokens "
            "on the same pages"
        )
    if tokens is not None:
        expected = int(tokens) // int(page_tokens)
        if int(tokens) % int(page_tokens) or expected != len(reference):
            raise ValueError(
                f"a materialized segment of {int(tokens)} tokens is "
                f"{expected} page(s) of {int(page_tokens)}, but "
                f"{len(reference)} Host page(s) were named"
            )
    return reference


@dataclass(frozen=True)
class RegionGeometry:
    """One Host region's per-token row: ``num_layers`` physical layers of
    ``dim`` elements of ``dtype`` (one head: MLA latent / MQA indexer key).

    ``layer_map`` maps an engine layer to its physical Host layer (-1 = the
    region keeps none for it); None is the identity.
    """

    num_layers: int
    dim: int
    dtype: torch.dtype
    layer_map: tuple[int, ...] | None = None

    def physical_layer(self, engine_layer: int, name: str) -> int:
        index = (
            int(engine_layer) if self.layer_map is None
            else int(self.layer_map[int(engine_layer)])
        )
        if not 0 <= index < int(self.num_layers):
            raise RuntimeError(
                f"engine layer {engine_layer} has no {name} Host layer "
                f"(resolved {index} of {self.num_layers})"
            )
        return index


@dataclass(frozen=True)
class DualRegionSlab:
    """One resident slab of a DSA segment: both regions, the same tokens.

    ``latent`` is ``[primary_layers, tokens, latent_dim]`` and ``indexer``
    ``[aux_layers, tokens, indexer_dim]``, indexed by PHYSICAL Host layer;
    both are views of the loader's staging, valid until the next slab.
    """

    token_start: int
    tokens: int
    latent: torch.Tensor
    indexer: torch.Tensor


class DualRegionSegmentLoader:
    """H2D staging for the materialized segments of one DSA wave.

    One staging buffer per region, both sized in the SAME number of Host
    pages so a slab is one token range in both. The two loads of a slab are
    launched back to back and both waited on before the slab is yielded --
    one wait point per slab, as in ``MaterializedSegmentLoader``, and no
    per-layer or per-token host synchronization.
    """

    def __init__(
        self,
        *,
        primary_view,
        aux_view,
        page_tokens: int,
        primary: RegionGeometry,
        aux: RegionGeometry,
        device,
        max_slab_bytes: int,
        torch_module=None,
    ) -> None:
        self._torch = torch_module if torch_module is not None else torch
        self._page_tokens = int(page_tokens)
        if self._page_tokens <= 0:
            raise ValueError(
                f"a dual-region loader needs a positive page size, got {page_tokens}"
            )
        self._regions = []
        page_bytes_all_layers = 0
        for name, view, geometry in (
            ("primary", primary_view, primary), ("aux", aux_view, aux),
        ):
            if view is None:
                raise ValueError(f"a dual-region loader needs the {name} Host view")
            if int(geometry.num_layers) <= 0 or int(geometry.dim) <= 0:
                raise ValueError(
                    f"the {name} region needs a positive layer count and row "
                    f"width, got layers={geometry.num_layers}, dim={geometry.dim}"
                )
            element = self._torch.empty((), dtype=geometry.dtype).element_size()
            page_bytes = self._page_tokens * int(geometry.dim) * element
            page_bytes_all_layers += page_bytes * int(geometry.num_layers)
            self._regions.append((name, view, geometry, page_bytes))
        slab_pages = int(max_slab_bytes) // page_bytes_all_layers
        if slab_pages < 1:
            raise ValueError(
                f"a staging budget of {int(max_slab_bytes)} B holds no page of "
                f"{page_bytes_all_layers} B across both Host regions"
            )
        self._slab_pages = slab_pages
        self._page_bytes_all_layers = page_bytes_all_layers
        self._staging = [
            self._torch.empty(
                (int(geometry.num_layers), slab_pages * self._page_tokens,
                 int(geometry.dim)),
                dtype=geometry.dtype,
                device=device,
            )
            for _, _, geometry, _ in self._regions
        ]

    @property
    def page_tokens(self) -> int:
        return self._page_tokens

    @property
    def primary(self) -> RegionGeometry:
        return self._regions[0][2]

    @property
    def aux(self) -> RegionGeometry:
        return self._regions[1][2]

    @property
    def slab_pages(self) -> int:
        return self._slab_pages

    @property
    def page_bytes_all_layers(self) -> int:
        """Host bytes of one page across every layer of BOTH regions."""

        return self._page_bytes_all_layers

    @property
    def staging_bytes(self) -> int:
        return sum(
            buffer.numel() * buffer.element_size() for buffer in self._staging
        )

    def _launch(self, view, staging, page_bytes, window):
        torch_module = self._torch
        count = len(window)
        base = staging.data_ptr()
        layer_stride = self._slab_pages * page_bytes
        pointers = torch_module.tensor(
            [
                [[
                    base + layer * layer_stride + page * page_bytes
                    for page in range(count)
                ]]
                for layer in range(int(staging.shape[0]))
            ],
            dtype=torch_module.int64,
        )
        return view.async_load_prefix_pages_to_device(
            host_page_ids=torch_module.tensor([window], dtype=torch_module.int64),
            active_page_counts=torch_module.tensor(
                [count], dtype=torch_module.int64
            ),
            k_device_ptrs=pointers,
            v_device_ptrs=None,
        )

    def slabs(self, *, pages_by_group, tokens: int) -> Iterator[DualRegionSlab]:
        """Load one DSA segment slab by slab; yields each slab once resident
        in BOTH regions' staging. The caller finishes a slab before the next.
        """

        pages = list(aligned_region_pages(
            pages_by_group, groups=(PRIMARY_GROUP, AUX_GROUP),
            page_tokens=self._page_tokens, tokens=tokens,
        ))
        for first in range(0, len(pages), self._slab_pages):
            window = pages[first:first + self._slab_pages]
            tasks = []
            try:
                for (_, view, _, page_bytes), staging in zip(
                    self._regions, self._staging
                ):
                    tasks.append(self._launch(view, staging, page_bytes, window))
            except Exception:
                # A launched load still writes the staging; drain it before
                # the error leaves, so nothing reuses memory a copy targets.
                for task in tasks:
                    task.wait()
                raise
            # Both copies are read by the pool copy in the next statement.
            for task in tasks:
                task.wait()
            slab_tokens = len(window) * self._page_tokens
            yield DualRegionSlab(
                token_start=first * self._page_tokens,
                tokens=slab_tokens,
                latent=self._staging[0][:, :slab_tokens, :],
                indexer=self._staging[1][:, :slab_tokens, :],
            )

    def release(self) -> None:
        self._staging = None

