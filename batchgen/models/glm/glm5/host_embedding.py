# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                         #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "License");              #
#  you may not use this file except in compliance with the License.              #
# ---------------------------------------------------------------------------- #

"""Host-resident ``embed_tokens`` for GLM-5.3.

GLM-5.3's ``[154880, 6144]`` bf16 embedding table is 1.7725 GiB. The shared
device skeleton (``Parallel_Strategy_Manager.share_skeleton_on_device``) used
to upload it once per rank, and both phase instances aliased that one device
copy, so every rank paid 1.7725 GiB of HBM for a table that a decode step
reads at most ``bucket`` rows of.

This module keeps the table in host memory and gathers only the step's rows:
``bucket`` x 6144 x 2 B, i.e. 1.5 MiB at bucket 128. The gather is a host
``index_select`` into a pinned staging buffer followed by one contiguous H2D
copy, which keeps the transfer on the copy engine instead of a device gather
kernel reading over PCIe.

Two deliberate choices, both load-bearing:

* **The table itself is never pinned.** It comes from the shared parameter-store
  mapping that all local worker processes read; pinning a shared/COW mapping
  privatizes its pages, so pinning here would turn one 1.7725 GiB host copy
  into one per rank. Only the small per-step staging buffer is pinned, which
  is what makes the H2D copy async-capable.
* **The weight is a plain attribute, not a Parameter or a registered buffer.**
  ``nn.Module.to(device)`` must not move it (the manager calls
  ``model.to(device)`` after binding), and
  ``share_skeleton_on_device``/``_skeleton_bind_keys`` derive the device upload
  set from ``model.named_parameters()`` -- staying out of that set is exactly
  what keeps the table off the device.

Forward numerics are bit-exact with ``nn.Embedding``: inference-time
``F.embedding`` is a pure row gather, and ``padding_idx`` affects only
gradients, so ``index_select`` reproduces it element for element.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HostResidentEmbedding(nn.Module):
    """``nn.Embedding``-compatible lookup whose table stays in host memory.

    Call contract matches ``nn.Embedding``: ``forward(input_ids)`` returns
    ``[*input_ids.shape, embedding_dim]`` on ``device``, for ``input_ids`` on
    either host or device.
    """

    # Duck-typing marker: the decode graph segment and its adapter test for
    # this instead of importing this class, so no graph code depends on where
    # the table lives beyond "it cannot be read inside a captured graph".
    host_resident = True

    def __init__(
        self,
        host_weight: torch.Tensor,
        device: torch.device,
        *,
        padding_idx: int | None = None,
        non_blocking: bool = False,
        max_pinned_rows: int = 1024,
    ) -> None:
        super().__init__()
        if host_weight.dim() != 2:
            raise ValueError(
                "HostResidentEmbedding requires a 2-D [vocab, hidden] table, "
                f"got shape {tuple(host_weight.shape)}"
            )
        if host_weight.device.type != "cpu":
            raise ValueError(
                "HostResidentEmbedding requires a host (CPU) table; got device "
                f"{host_weight.device}. Uploading it defeats the whole point."
            )
        # object.__setattr__ bypasses nn.Module's Parameter/buffer registration
        # on purpose: see the module docstring.
        object.__setattr__(self, "host_weight", host_weight)
        self.num_embeddings = int(host_weight.shape[0])
        self.embedding_dim = int(host_weight.shape[1])
        self.padding_idx = None if padding_idx is None else int(padding_idx)
        self.target_device = torch.device(device)
        self.non_blocking = bool(non_blocking)
        self.max_pinned_rows = int(max_pinned_rows)
        self._staging: torch.Tensor | None = None

    # ``embed.weight.shape`` is read by callers that only want the table's
    # geometry (vocab size); expose it without registering a Parameter.
    @property
    def weight(self) -> torch.Tensor:
        return self.host_weight

    def host_bytes(self) -> int:
        return self.host_weight.numel() * self.host_weight.element_size()

    def _staging_rows(self, rows: int) -> torch.Tensor:
        """``[rows, hidden]`` staging view for the gather.

        A decode step asks for at most one row per sequence, so the reusable
        buffer is pinned and the H2D copy can go async. A prefill micro-batch
        asks for one row per prompt token, which at 6144 x 2 B per row is
        hundreds of MiB; keeping a pinned allocation that large resident would
        cost far more than the copy it accelerates, so past
        ``max_pinned_rows`` the gather uses a throwaway unpinned buffer.
        """
        if rows > self.max_pinned_rows:
            return torch.empty(
                rows, self.embedding_dim, dtype=self.host_weight.dtype
            )
        if self._staging is None or self._staging.shape[0] < rows:
            self._staging = torch.empty(
                self.max_pinned_rows,
                self.embedding_dim,
                dtype=self.host_weight.dtype,
                pin_memory=torch.cuda.is_available(),
            )
        return self._staging[:rows]

    def gather_host_rows(self, ids_host: torch.Tensor) -> torch.Tensor:
        """Gather ``ids_host`` (1-D, CPU, int64) into the pinned staging view."""
        if ids_host.device.type != "cpu":
            raise ValueError(
                f"gather_host_rows needs CPU ids, got {ids_host.device}"
            )
        flat = ids_host.reshape(-1)
        if flat.numel() and (
            int(flat.min()) < 0 or int(flat.max()) >= self.num_embeddings
        ):
            raise IndexError(
                f"token id out of range for vocab {self.num_embeddings}: "
                f"[{int(flat.min())}, {int(flat.max())}]"
            )
        rows = self._staging_rows(flat.numel())
        if flat.numel():
            torch.index_select(
                self.host_weight, 0, flat.to(torch.int64), out=rows
            )
        return rows

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        ids_host = (
            input_ids
            if input_ids.device.type == "cpu"
            else input_ids.detach().to("cpu")
        )
        rows = self.gather_host_rows(ids_host)
        out = rows.to(device=self.target_device, non_blocking=self.non_blocking)
        return out.view(*tuple(input_ids.shape), self.embedding_dim)

    def extra_repr(self) -> str:
        return (
            f"{self.num_embeddings}, {self.embedding_dim}, host_resident=True, "
            f"host_gib={self.host_bytes() / (1024 ** 3):.4f}"
        )
