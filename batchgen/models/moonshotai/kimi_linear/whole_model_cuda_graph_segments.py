# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "License");              #
#  you may not use this file except in compliance with the License.              #
# ---------------------------------------------------------------------------- #

"""Whole-model decode graph for Kimi-K3.

The existing K3 graph driver captures one attention span and one resident-MoE
segment at a time.  That removes kernel setup inside a layer, but Python still
returns to the driver between all 93 layers.  This segment composes the same
already-validated child forwards into one graph:

    embedding -> [attention span + resident grouped MoE] x 93
               -> block-attention-residual output mix -> final RMSNorm

The language-model head intentionally remains outside this segment.  The
worker already owns token selection, and keeping the head outside avoids
capturing a vocabulary-sized output buffer on every rank.

K3's TP8 attention groups replicate the batch before the resident EP path
scatters it.  A graph bucket is global-EP safe, but an admitted batch can be
smaller than that bucket and need a non-divisible TP8 split.  The integer maps
below encode every possible valid row count for a bucket.  The graph selects a
row-count map on device from ``num_valid_tokens``; no host ``.item()`` or
per-layer Python scatter is needed during replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import torch

from batchgen.cuda_graph.graph_manager import TensorSpec
from .moe_tp_reshard import balanced_row_split


@dataclass
class _WholeBucketMaps:
    """Device-resident row maps for one model/TP bucket."""

    padded_indices: torch.Tensor
    padded_valid: torch.Tensor
    local_indices: torch.Tensor
    local_valid: torch.Tensor
    local_counts: torch.Tensor
    original_indices: torch.Tensor
    original_valid: torch.Tensor
    group_bucket: int
    local_bucket: int


class KimiLinearWholeModelSegment:
    """Graph-capturable K3 transformer body with inline resident MoE."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        layer_segments: Iterable[object],
        moe_segments: Mapping[int, object],
        statics: Dict[int, object],
        tp_size: int,
        tp_rank: int,
        device: torch.device,
        hidden_size: int,
        dtype: torch.dtype,
        max_bucket_size: int,
    ) -> None:
        if tp_size <= 0 or not 0 <= tp_rank < tp_size:
            raise ValueError(
                f"invalid TP coordinates tp_size={tp_size}, tp_rank={tp_rank}"
            )
        self.model = model
        self.layer_segments = list(layer_segments)
        if not self.layer_segments:
            raise ValueError("KimiLinearWholeModelSegment needs decoder layers")
        self.moe_segments = dict(moe_segments)
        self.statics = statics
        self.tp_size = int(tp_size)
        self.tp_rank = int(tp_rank)
        self.device = device
        self.hidden_size = int(hidden_size)
        self.dtype = dtype
        self.max_bucket_size = int(max_bucket_size)
        self.num_layers = len(self.layer_segments)

        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None or len(layers) != self.num_layers:
            raise ValueError(
                "KimiLinearWholeModelSegment layer count does not match model"
            )

        self._bucket_maps: Dict[int, _WholeBucketMaps] = {}
        # The graph keeps the model's logical layer numbering, but only MLA
        # layers produce a paged-KV row.  Keep that mapping local to the graph
        # so the post-replay staging copy does not clone 69 never-written KDA
        # rows on every decode token.
        self._primary_kv_layers = tuple(
            layer_idx
            for layer_idx, segment in enumerate(self.layer_segments)
            if not bool(getattr(segment, "is_kda", False))
        )
        self._logical_to_physical_kv = tuple(
            {
                layer_idx: physical_idx
                for physical_idx, layer_idx in enumerate(self._primary_kv_layers)
            }.get(layer_idx, -1)
            for layer_idx in range(self.num_layers)
        )
        self._kv_key_buffer: Optional[torch.Tensor] = None
        self._kv_buffers: Optional[list[dict[str, torch.Tensor | None]]] = None
        self.primary_kv_offload_buffers = None
        self._no_v_cache = True
        self._primary_kv_dim = 0
        for segment in self.layer_segments:
            if not bool(getattr(segment, "is_kda", False)):
                self._primary_kv_dim = int(getattr(segment, "kv_dim", 0))
                if self._primary_kv_dim > 0:
                    break

    # ------------------------------------------------------------------ #
    # CapturableSegment protocol                                         #
    # ------------------------------------------------------------------ #

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "input_ids": TensorSpec(("batch_size", 1), torch.int64),
            # This is the local rank's real row count.  The graph bucket is
            # selected from the globally synchronized maximum, so this scalar
            # may be smaller on a DP group with fewer admitted sequences.
            "num_valid_tokens": TensorSpec((1,), torch.int32),
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "hidden_states": TensorSpec(
                ("batch_size", 1, self.hidden_size), self.dtype
            )
        }

    def setup_static_buffers(self, bucket_size: int) -> None:
        bucket = int(bucket_size)
        if bucket <= 0 or bucket > self.max_bucket_size:
            raise ValueError(
                f"whole-model bucket {bucket} is outside 1..{self.max_bucket_size}"
            )
        if bucket not in self._bucket_maps:
            self._bucket_maps[bucket] = self._make_bucket_maps(bucket)

        # Child attention spans bind the bucket-owned KV/KDA/block-residual
        # statics.  The same objects are then called directly from this outer
        # segment, never through a nested graph replay.
        for segment in self.layer_segments:
            setup = getattr(segment, "setup_static_buffers", None)
            if setup is not None:
                setup(bucket)

        # One contiguous staging buffer covers only the MLA layers.  It is
        # kept for the lifetime of the outer segment so dropping one bucket
        # cannot invalidate another captured graph's pointer.  KDA layers do
        # not emit paged KV and must not consume staging capacity.
        if self._kv_key_buffer is None and self._primary_kv_dim > 0:
            self._kv_key_buffer = torch.zeros(
                len(self._primary_kv_layers),
                self.max_bucket_size,
                1,
                1,
                self._primary_kv_dim,
                dtype=self.dtype,
                device=self.device,
            )
            self._kv_buffers = [
                {"key": self._kv_key_buffer[i], "value": None}
                for i in range(len(self._primary_kv_layers))
            ]
            self.primary_kv_offload_buffers = self._kv_buffers

    def release_static_buffers(self, bucket_size: int) -> None:
        # Do not release the shared KV staging buffer here: CUDAGraphManager
        # calls this method once per dropped bucket, while another bucket may
        # still hold a graph that captured the same outer segment.
        for segment in self.layer_segments:
            release = getattr(segment, "release_static_buffers", None)
            if release is not None:
                release(bucket_size)
        self._bucket_maps.pop(int(bucket_size), None)

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        num_valid_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        bucket = int(input_ids.shape[0])
        maps = self._bucket_maps[bucket]
        selector = num_valid_tokens.reshape(()).to(torch.long).clamp(
            min=0, max=bucket
        )

        hidden_states = self.model.model.embed_tokens(input_ids)
        valid = (
            torch.arange(bucket, dtype=torch.int32, device=hidden_states.device)
            < num_valid_tokens.reshape(1)
        )
        # Embedding id 0 is a real vocabulary entry, not guaranteed zero.  Mask
        # graph padding before the first KDA/MLA layer rather than relying on a
        # padding token convention.
        hidden_states = hidden_states * valid.view(bucket, 1, 1).to(
            hidden_states.dtype
        )

        for layer_idx, layer_segment in enumerate(self.layer_segments):
            outputs = layer_segment.forward(hidden_states)
            k_tensor = outputs.get("k_tensor")
            if k_tensor is not None and self._kv_key_buffer is not None:
                self._copy_primary_kv(layer_idx, k_tensor)

            if bool(getattr(layer_segment, "fold_ffn", False)):
                hidden_states = outputs["hidden"]
                continue

            moe = self.moe_segments.get(layer_idx)
            if moe is None:
                raise RuntimeError(
                    f"Kimi-K3 whole graph has no resident MoE for layer {layer_idx}"
                )

            hidden_2d = outputs["normed"].reshape(bucket, self.hidden_size)
            # K3's resident EP input uses rank-major TP-group rows.  The maps
            # implement the same balanced split as the eager path, including
            # underfilled/non-divisible batches, entirely on device.
            padded_map = maps.padded_indices[selector]
            padded_valid = maps.padded_valid[selector]
            padded = hidden_2d.index_select(0, padded_map)
            padded = padded * padded_valid.to(hidden_2d.dtype).unsqueeze(-1)

            local_map = maps.local_indices[selector]
            local_valid = maps.local_valid[selector]
            local = hidden_2d.index_select(0, local_map)
            local = local * local_valid.to(hidden_2d.dtype).unsqueeze(-1)
            local_count = maps.local_counts.index_select(0, selector.reshape(1))

            moe_output = moe.forward(
                padded=padded,
                local=local,
                num_valid_tokens=local_count,
            )["moe_output"]
            original_map = maps.original_indices[selector]
            original_valid = maps.original_valid[selector]
            reassembled = moe_output.index_select(0, original_map)
            reassembled = reassembled * original_valid.to(
                reassembled.dtype
            ).unsqueeze(-1)
            hidden_states = outputs["residual"] + reassembled.view_as(
                outputs["residual"]
            )

        if bool(getattr(self.model.model, "use_attn_residuals", False)):
            statics = self.statics[bucket]
            mixed = self.model.model._apply_output_attn_res(
                hidden_states.reshape(-1, self.hidden_size),
                statics.block_residual,
            )
            hidden_states = mixed.view_as(hidden_states)
        hidden_states = self.model.model.norm(hidden_states)
        hidden_states = hidden_states * valid.view(bucket, 1, 1).to(
            hidden_states.dtype
        )
        return {"hidden_states": hidden_states}

    def _copy_primary_kv(self, layer_idx: int, k_tensor: torch.Tensor) -> None:
        if self._kv_key_buffer is None:
            raise RuntimeError("K3 whole graph KV staging is not initialized")
        physical_layer_idx = self._logical_to_physical_kv[int(layer_idx)]
        if physical_layer_idx < 0:
            raise KeyError(
                f"logical KDA layer {layer_idx} does not own paged KV"
            )
        if k_tensor.dim() == 3:
            k_tensor = k_tensor.unsqueeze(2)
        expected = (k_tensor.shape[0], 1, 1, self._primary_kv_dim)
        if tuple(k_tensor.shape) != expected:
            raise RuntimeError(
                f"K3 whole graph KV tensor mismatch: got {tuple(k_tensor.shape)}, "
                f"expected (*, 1, 1, {self._primary_kv_dim})"
            )
        self._kv_key_buffer[physical_layer_idx, : k_tensor.shape[0]].copy_(k_tensor)

    # ------------------------------------------------------------------ #
    # TP row-map construction                                             #
    # ------------------------------------------------------------------ #

    def _make_bucket_maps(self, bucket: int) -> _WholeBucketMaps:
        local_bucket = (bucket + self.tp_size - 1) // self.tp_size
        group_bucket = self.tp_size * local_bucket

        # Construct the lookup tables on the CPU and transfer each complete
        # table once.  The old version performed one CUDA write per map entry
        # (O(bucket**2 * TP)) during setup; for a startup capture this created
        # thousands of tiny host launches and could also leave a partially
        # initialized device table visible if setup was interrupted.
        padded_indices_cpu = torch.zeros(
            (bucket + 1, group_bucket), dtype=torch.long
        )
        padded_valid_cpu = torch.zeros(
            (bucket + 1, group_bucket), dtype=torch.bool
        )
        local_indices_cpu = torch.zeros(
            (bucket + 1, local_bucket), dtype=torch.long
        )
        local_valid_cpu = torch.zeros(
            (bucket + 1, local_bucket), dtype=torch.bool
        )
        local_counts_cpu = torch.zeros((bucket + 1,), dtype=torch.int32)
        original_indices_cpu = torch.zeros(
            (bucket + 1, bucket), dtype=torch.long
        )
        original_valid_cpu = torch.zeros(
            (bucket + 1, bucket), dtype=torch.bool
        )

        # These are setup-time CPU loops.  All tensors selected by ``forward``
        # are already on-device before capture begins.
        for valid_rows in range(bucket + 1):
            splits = balanced_row_split(valid_rows, self.tp_size)
            for group_rank, (start, end) in enumerate(splits):
                for local_pos, row in enumerate(range(start, end)):
                    rank_major_pos = group_rank * local_bucket + local_pos
                    padded_indices_cpu[valid_rows, rank_major_pos] = row
                    padded_valid_cpu[valid_rows, rank_major_pos] = True
                    if group_rank == self.tp_rank:
                        local_indices_cpu[valid_rows, local_pos] = row
                        local_valid_cpu[valid_rows, local_pos] = True
                    original_indices_cpu[valid_rows, row] = rank_major_pos
                    original_valid_cpu[valid_rows, row] = True
            local_counts_cpu[valid_rows] = (
                splits[self.tp_rank][1] - splits[self.tp_rank][0]
            )

        def to_device(table: torch.Tensor) -> torch.Tensor:
            return table.to(device=self.device, non_blocking=True)

        return _WholeBucketMaps(
            padded_indices=to_device(padded_indices_cpu),
            padded_valid=to_device(padded_valid_cpu),
            local_indices=to_device(local_indices_cpu),
            local_valid=to_device(local_valid_cpu),
            local_counts=to_device(local_counts_cpu),
            original_indices=to_device(original_indices_cpu),
            original_valid=to_device(original_valid_cpu),
            group_bucket=group_bucket,
            local_bucket=local_bucket,
        )


__all__ = ["KimiLinearWholeModelSegment"]
