"""Prefill Host KV offload helpers."""

from __future__ import annotations

from typing import Callable, List, Optional

import torch

from batchgen.models.wrappers.prefix_cache import (
    PrefixCachePrepackMetadata,
    ensure_prefix_cache_prepack_metadata,
)


class PrefillHostKVOffloader:
    """Offload prepacked KV with optional destination offsets."""

    def __init__(
        self,
        *,
        worker_view: object,
        layer_idx: int,
        metadata: PrefixCachePrepackMetadata,
        track_task: Optional[Callable[[object, int], None]] = None,
        pin_tensor: Optional[Callable[[torch.Tensor, int], None]] = None,
    ):
        if worker_view is None:
            raise RuntimeError("Prefill offload requires host KV view")
        self.worker_view = worker_view
        self.layer_idx = int(layer_idx)
        self.metadata = ensure_prefix_cache_prepack_metadata(metadata)
        self.track_task = track_task
        self.pin_tensor = pin_tensor

    def _track(self, task: object) -> None:
        if task is not None and self.track_task is not None:
            self.track_task(task, self.layer_idx)

    def _pin(self, tensor: torch.Tensor) -> None:
        if self.pin_tensor is not None:
            self.pin_tensor(tensor, self.layer_idx)

    def _pin_parent_tensors(self, *tensors: torch.Tensor) -> None:
        should_sync = False
        for tensor in tensors:
            self._pin(tensor)
            should_sync = should_sync or bool(getattr(tensor, "is_cuda", False))
        if should_sync:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
            event.synchronize()

    def _destination_starts(self) -> Optional[List[int]]:
        if not self.metadata.prefix_reuse_mode:
            return None
        if not hasattr(
            self.worker_view, "async_offload_layer_kv_to_host_with_offsets"
        ):
            raise RuntimeError(
                "Prefill offset offload requires "
                "async_offload_layer_kv_to_host_with_offsets"
            )
        return [int(tokens) for tokens in self.metadata.prefix_shared_tokens]

    def _offload_one(
        self,
        *,
        sequence_id: int,
        k_tensor: torch.Tensor,
        v_tensor: Optional[torch.Tensor],
        sequence_length: int,
        destination_start: Optional[int],
    ) -> None:
        if destination_start is None:
            task = self.worker_view.async_offload_layer_kv_to_host(
                layer_idx=self.layer_idx,
                sequence_ids=[int(sequence_id)],
                k_tensor=k_tensor,
                v_tensor=v_tensor,
                sequence_lengths=[int(sequence_length)],
            )
        else:
            task = self.worker_view.async_offload_layer_kv_to_host_with_offsets(
                layer_idx=self.layer_idx,
                sequence_ids=[int(sequence_id)],
                k_tensor=k_tensor,
                v_tensor=v_tensor,
                sequence_lengths=[int(sequence_length)],
                source_token_starts=[0],
                destination_token_starts=[int(destination_start)],
            )
        self._track(task)

    def offload_gqa(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        sequence_callback: Optional[
            Callable[[int, int, int, torch.Tensor, torch.Tensor], None]
        ] = None,
    ) -> None:
        self._pin_parent_tensors(key, value)
        cu = self.metadata.cu_seqlens_list()
        destination_starts = self._destination_starts()
        for seq_idx, sequence_id in enumerate(
            self.metadata.global_sequence_ids
        ):
            start_idx = int(cu[seq_idx])
            end_idx = int(cu[seq_idx + 1])
            seq_len = end_idx - start_idx
            seq_key = key[start_idx:end_idx].unsqueeze(0)
            seq_value = value[start_idx:end_idx].unsqueeze(0)
            self._pin(seq_key)
            self._pin(seq_value)
            if sequence_callback is not None:
                sequence_callback(
                    seq_idx, sequence_id, seq_len, seq_key, seq_value
                )
            self._offload_one(
                sequence_id=sequence_id,
                k_tensor=seq_key,
                v_tensor=seq_value,
                sequence_length=seq_len,
                destination_start=(
                    None
                    if destination_starts is None
                    else destination_starts[seq_idx]
                ),
            )

    def offload_mla(
        self,
        *,
        key: torch.Tensor,
        sequence_callback: Optional[
            Callable[[int, int, int, torch.Tensor], None]
        ] = None,
    ) -> None:
        self._pin_parent_tensors(key)
        cu = self.metadata.cu_seqlens_list()
        destination_starts = self._destination_starts()
        for seq_idx, sequence_id in enumerate(
            self.metadata.global_sequence_ids
        ):
            start_idx = int(cu[seq_idx])
            end_idx = int(cu[seq_idx + 1])
            seq_len = end_idx - start_idx
            seq_key = key[start_idx:end_idx]
            if seq_key.dim() == 2:
                seq_key = seq_key.unsqueeze(0).unsqueeze(2)
            elif seq_key.dim() == 3:
                seq_key = seq_key.unsqueeze(0)
            else:
                raise RuntimeError(
                    "MLA prefill offload expects 2D or 3D KV, "
                    f"got {seq_key.dim()}D"
                )
            self._pin(seq_key)
            if sequence_callback is not None:
                sequence_callback(seq_idx, sequence_id, seq_len, seq_key)
            self._offload_one(
                sequence_id=sequence_id,
                k_tensor=seq_key,
                v_tensor=None,
                sequence_length=seq_len,
                destination_start=(
                    None
                    if destination_starts is None
                    else destination_starts[seq_idx]
                ),
            )
