"""Common prefix-cache helpers for model attention wrappers."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch


def _build_cu_seqlens_values(seq_lengths: Sequence[int]) -> List[int]:
    values = [0]
    running = 0
    for length in seq_lengths:
        running += int(length)
        values.append(running)
    return values


def ensure_prefix_cache_prepack_metadata(
    metadata,
) -> "PrefixCachePrepackMetadata":
    """Normalize explicit or legacy-compatible prefix metadata."""

    if isinstance(metadata, PrefixCachePrepackMetadata):
        return metadata
    if getattr(metadata, "phase", None) is not None:
        return PrefixCachePrepackMetadata.from_forward_metadata(metadata)
    if getattr(metadata, "cu_seqlens_q", None) is not None:
        prefix_reuse = getattr(metadata, "prefix_reuse", None)
        if prefix_reuse is None:
            raise RuntimeError(
                "PrefillAttentionMetadata without prefix reuse does not carry "
                "global sequence ids; pass ForwardBatchMetadata instead"
            )
        return PrefixCachePrepackMetadata.from_prefill_metadata(
            metadata,
            global_sequence_ids=prefix_reuse.global_sequence_ids,
        )
    raise TypeError(
        "metadata must be PrefixCachePrepackMetadata, PrefillAttentionMetadata, "
        "or ForwardBatchMetadata"
    )


@dataclass(frozen=True)
class PrefixCachePrepackMetadata:
    """Validated prepack metadata needed by prefix-cache-aware wrappers."""

    cu_seqlens: torch.Tensor
    cu_seqlens_cpu: List[int]
    max_seqlen: int
    num_sequences: int
    seq_lengths: List[int]
    global_sequence_ids: List[int]
    prefix_reuse_mode: bool
    full_hit_mode: bool
    prefix_shared_tokens: Optional[List[int]]
    full_seq_lengths: Optional[List[int]]

    @classmethod
    def from_prefill_metadata(
        cls,
        prefill_metadata,
        *,
        global_sequence_ids: Sequence[int],
    ) -> "PrefixCachePrepackMetadata":
        """Build wrapper-compatible metadata from explicit prefill metadata."""

        prefix_reuse = prefill_metadata.prefix_reuse
        prefix_shared_tokens = None
        full_seq_lengths = None
        prefix_reuse_mode = False
        full_hit_mode = False
        seq_lengths = [int(length) for length in prefill_metadata.q_seq_lens]
        if prefix_reuse is not None:
            full_seq_lengths = [
                int(length) for length in prefill_metadata.kv_seq_lens
            ]
            prefix_shared_tokens = [
                int(full_len) - int(query_len)
                for query_len, full_len in zip(seq_lengths, full_seq_lengths)
            ]
            prefix_reuse_mode = any(
                tokens > 0 for tokens in prefix_shared_tokens
            )
            full_hit_mode = False

        metadata = cls(
            cu_seqlens=prefill_metadata.cu_seqlens_q,
            cu_seqlens_cpu=_build_cu_seqlens_values(seq_lengths),
            max_seqlen=int(prefill_metadata.max_seqlen_q),
            num_sequences=int(prefill_metadata.batch_size),
            seq_lengths=seq_lengths,
            global_sequence_ids=[int(seq_id) for seq_id in global_sequence_ids],
            prefix_reuse_mode=prefix_reuse_mode,
            full_hit_mode=full_hit_mode,
            prefix_shared_tokens=prefix_shared_tokens,
            full_seq_lengths=full_seq_lengths,
        )
        return metadata

    @classmethod
    def from_forward_metadata(
        cls,
        forward_metadata,
    ) -> "PrefixCachePrepackMetadata":
        """Build wrapper-compatible metadata from a bound forward metadata object."""

        if (
            forward_metadata.phase != "prefill"
            or forward_metadata.prefill is None
        ):
            raise RuntimeError(
                "Prefix cache prepack metadata requires bound prefill metadata"
            )
        return cls.from_prefill_metadata(
            forward_metadata.prefill,
            global_sequence_ids=forward_metadata.global_sequence_ids,
        )

    @classmethod
    def from_wrapper_cls(
        cls, wrapper_cls: type
    ) -> "PrefixCachePrepackMetadata":
        """Build metadata from legacy wrapper class variables."""

        cu_seqlens = getattr(wrapper_cls, "prepack_cu_seqlens", None)
        max_seqlen = getattr(wrapper_cls, "prepack_max_seqlen", None)
        num_sequences = getattr(wrapper_cls, "prepack_num_sequences", None)
        seq_lengths = getattr(wrapper_cls, "prepack_seq_lengths", None)
        global_sequence_ids = getattr(wrapper_cls, "cur_batch", None)
        prefix_reuse_mode = bool(
            getattr(wrapper_cls, "prepack_prefix_reuse_mode", False)
        )
        full_hit_mode = bool(
            getattr(wrapper_cls, "prepack_full_hit_mode", False)
        )
        if full_hit_mode:
            raise RuntimeError(
                "Legacy full-hit prefix mode is no longer supported; "
                "planner must clamp full hits to one-token extend prefill"
            )
        prefix_shared_tokens = getattr(
            wrapper_cls, "prepack_prefix_shared_tokens", None
        )
        full_seq_lengths = getattr(
            wrapper_cls, "prepack_full_seq_lengths", None
        )

        if cu_seqlens is None:
            raise RuntimeError(
                "Prefix cache prepack metadata requires cu_seqlens"
            )
        if max_seqlen is None:
            raise RuntimeError(
                "Prefix cache prepack metadata requires max_seqlen"
            )
        if num_sequences is None:
            raise RuntimeError(
                "Prefix cache prepack metadata requires num_sequences"
            )
        if seq_lengths is None:
            raise RuntimeError(
                "Prefix cache prepack metadata requires seq_lengths"
            )
        if global_sequence_ids is None:
            raise RuntimeError(
                "Prefix cache prepack metadata requires cur_batch"
            )

        seq_lengths = [int(length) for length in seq_lengths]
        global_sequence_ids = [int(seq_id) for seq_id in global_sequence_ids]
        num_sequences = int(num_sequences)
        if len(seq_lengths) != num_sequences:
            raise RuntimeError(
                "Prefix cache seq_lengths length does not match num_sequences: "
                f"{len(seq_lengths)} != {num_sequences}"
            )
        if len(global_sequence_ids) != num_sequences:
            raise RuntimeError(
                "Prefix cache cur_batch length does not match num_sequences: "
                f"{len(global_sequence_ids)} != {num_sequences}"
            )
        if len(cu_seqlens) != num_sequences + 1:
            raise RuntimeError(
                "Prefix cache cu_seqlens length does not match num_sequences: "
                f"{len(cu_seqlens)} != {num_sequences + 1}"
            )

        needs_prefix_metadata = prefix_reuse_mode
        if needs_prefix_metadata:
            if prefix_shared_tokens is None:
                raise RuntimeError(
                    "Prefix cache mode requires prepack_prefix_shared_tokens"
                )
            if full_seq_lengths is None:
                raise RuntimeError(
                    "Prefix cache mode requires prepack_full_seq_lengths"
                )
            prefix_shared_tokens = [
                int(tokens) for tokens in prefix_shared_tokens
            ]
            full_seq_lengths = [int(length) for length in full_seq_lengths]
            if len(prefix_shared_tokens) != num_sequences:
                raise RuntimeError(
                    "Prefix shared token count length does not match batch: "
                    f"{len(prefix_shared_tokens)} != {num_sequences}"
                )
            if len(full_seq_lengths) != num_sequences:
                raise RuntimeError(
                    "Full sequence length metadata length does not match batch: "
                    f"{len(full_seq_lengths)} != {num_sequences}"
                )
            for idx, (query_len, prefix_tokens, full_length) in enumerate(
                zip(seq_lengths, prefix_shared_tokens, full_seq_lengths)
            ):
                expected_full_length = int(query_len) + int(prefix_tokens)
                if expected_full_length != int(full_length):
                    raise RuntimeError(
                        "Prefix cache full length mismatch at sequence "
                        f"{idx}: query={query_len}, prefix={prefix_tokens}, "
                        f"full={full_length}"
                    )

        metadata = cls(
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=_build_cu_seqlens_values(seq_lengths),
            max_seqlen=int(max_seqlen),
            num_sequences=num_sequences,
            seq_lengths=seq_lengths,
            global_sequence_ids=global_sequence_ids,
            prefix_reuse_mode=prefix_reuse_mode,
            full_hit_mode=full_hit_mode,
            prefix_shared_tokens=prefix_shared_tokens,
            full_seq_lengths=full_seq_lengths,
        )
        return metadata

    def cu_seqlens_list(self) -> List[int]:
        return list(self.cu_seqlens_cpu)

    def sequence_span(self, seq_idx: int) -> Tuple[int, int]:
        cu = self.cu_seqlens_list()
        return cu[seq_idx], cu[seq_idx + 1]


class HostPrefixPageReader:
    """Read cached host KV pages for prefix-cache attention replay."""

    def __init__(
        self, *, core_engine: object, engine_config: object, layer_idx: int
    ):
        self.core_engine = core_engine
        self.engine_config = engine_config
        self.layer_idx = int(layer_idx)

    def page_size(self) -> int:
        host_cfg = getattr(self.engine_config, "Host_Paged_KV_Config", None)
        if host_cfg is None:
            host_cfg = getattr(self.engine_config, "host_paged_kv_config", None)
        if host_cfg is None or not hasattr(host_cfg, "page_size"):
            raise RuntimeError(
                "Prefix cache requires Host_Paged_KV_Config.page_size"
            )
        return int(host_cfg.page_size)

    def worker_view(self) -> object:
        worker_view = getattr(
            self.core_engine, "host_paged_kv_worker_view", None
        )
        if worker_view is None:
            raise RuntimeError(
                "Prefix cache requires host_paged_kv_worker_view"
            )
        return worker_view

    def _load_tensor(
        self,
        page_ptrs: Sequence[int],
        num_tokens: int,
        *,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        num_tokens = int(num_tokens)
        num_heads = int(num_heads)
        head_dim = int(head_dim)
        if num_tokens == 0:
            return torch.empty(
                (0, num_heads, head_dim), dtype=dtype, device=device
            )
        if dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError(
                f"Prefix cache host KV loader supports 16-bit KV only, got {dtype}"
            )

        page_size = self.page_size()
        elems_per_page = page_size * num_heads * head_dim
        remaining = num_tokens
        chunks = []
        for ptr in page_ptrs:
            if remaining <= 0:
                break
            take = min(page_size, remaining)
            array_type = ctypes.c_uint16 * elems_per_page
            host_array = array_type.from_address(int(ptr))
            host_uint16 = torch.frombuffer(host_array, dtype=torch.uint16)
            page_tensor = host_uint16.view(dtype).reshape(
                page_size, num_heads, head_dim
            )
            chunks.append(page_tensor[:take].clone())
            remaining -= take

        if remaining != 0:
            raise RuntimeError(
                "Host prefix KV page list is short by "
                f"{remaining} tokens (requested={num_tokens})"
            )

        return torch.cat(chunks, dim=0).to(
            device=device, dtype=dtype, non_blocking=True
        )

    def _sequence_layer_page_pointers(
        self, sequence_id: int, num_tokens: int
    ) -> Tuple[List[int], Optional[List[int]]]:
        return self.worker_view().get_sequence_layer_page_pointers(
            int(sequence_id),
            self.layer_idx,
            int(num_tokens),
        )

    def load_gqa_kv(
        self,
        sequence_id: int,
        num_tokens: int,
        *,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k_ptrs, v_ptrs = self._sequence_layer_page_pointers(
            sequence_id, num_tokens
        )
        if v_ptrs is None:
            raise RuntimeError("GQA prefix cache requires host V cache pages")
        return (
            self._load_tensor(
                list(k_ptrs),
                num_tokens,
                num_heads=num_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
            ),
            self._load_tensor(
                list(v_ptrs),
                num_tokens,
                num_heads=num_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
            ),
        )

    def load_mla_kv(
        self,
        sequence_id: int,
        num_tokens: int,
        *,
        kv_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        k_ptrs, _ = self._sequence_layer_page_pointers(sequence_id, num_tokens)
        return self._load_tensor(
            list(k_ptrs),
            num_tokens,
            num_heads=1,
            head_dim=kv_dim,
            dtype=dtype,
            device=device,
        )


class PrefixAttentionKvBuilder:
    """Build varlen attention KV tensors from cached prefix and suffix KV."""

    def __init__(self, reader: HostPrefixPageReader):
        self.reader = reader

    def build_gqa_prefix_kv(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: PrefixCachePrepackMetadata,
        num_heads: int,
        head_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if metadata.prefix_shared_tokens is None:
            raise RuntimeError(
                "GQA prefix KV build requires prefix token metadata"
            )

        device = key.device
        cu_cpu = metadata.cu_seqlens_list()
        k_segments = []
        v_segments = []
        cu_k = [0]
        max_seqlen_k = 0

        for seq_idx, suffix_len in enumerate(metadata.seq_lengths):
            start_idx = int(cu_cpu[seq_idx])
            end_idx = int(cu_cpu[seq_idx + 1])
            prefix_tokens = int(metadata.prefix_shared_tokens[seq_idx])
            suffix_k = key[start_idx:end_idx]
            suffix_v = value[start_idx:end_idx]
            if prefix_tokens > 0:
                prefix_k, prefix_v = self.reader.load_gqa_kv(
                    metadata.global_sequence_ids[seq_idx],
                    prefix_tokens,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dtype=key.dtype,
                    device=device,
                )
                seq_k = torch.cat([prefix_k, suffix_k], dim=0)
                seq_v = torch.cat([prefix_v, suffix_v], dim=0)
            else:
                seq_k = suffix_k
                seq_v = suffix_v

            k_segments.append(seq_k)
            v_segments.append(seq_v)
            cu_k.append(cu_k[-1] + int(seq_k.shape[0]))
            max_seqlen_k = max(max_seqlen_k, int(seq_k.shape[0]))

        return (
            torch.cat(k_segments, dim=0),
            torch.cat(v_segments, dim=0),
            torch.tensor(cu_k, dtype=torch.int32, device=device),
            max_seqlen_k,
        )

    def build_mla_prefix_kv(
        self,
        *,
        key: torch.Tensor,
        metadata: PrefixCachePrepackMetadata,
        kv_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if metadata.prefix_shared_tokens is None:
            raise RuntimeError(
                "MLA prefix KV build requires prefix token metadata"
            )

        device = key.device
        cu_cpu = metadata.cu_seqlens_list()
        k_segments = []
        cu_k = [0]
        max_seqlen_k = 0

        for seq_idx, suffix_len in enumerate(metadata.seq_lengths):
            start_idx = int(cu_cpu[seq_idx])
            end_idx = int(cu_cpu[seq_idx + 1])
            prefix_tokens = int(metadata.prefix_shared_tokens[seq_idx])
            suffix_k = key[start_idx:end_idx]
            if suffix_k.dim() == 2:
                suffix_k = suffix_k.unsqueeze(1)
            if prefix_tokens > 0:
                prefix_k = self.reader.load_mla_kv(
                    metadata.global_sequence_ids[seq_idx],
                    prefix_tokens,
                    kv_dim=kv_dim,
                    dtype=key.dtype,
                    device=device,
                )
                seq_k = torch.cat([prefix_k, suffix_k], dim=0)
            else:
                seq_k = suffix_k

            k_segments.append(seq_k)
            cu_k.append(cu_k[-1] + int(seq_k.shape[0]))
            max_seqlen_k = max(max_seqlen_k, int(seq_k.shape[0]))

        return (
            torch.cat(k_segments, dim=0),
            torch.tensor(cu_k, dtype=torch.int32, device=device),
            max_seqlen_k,
        )


class PrefixAwarePrefillOffloader:
    """Offload prepacked KV with optional prefix-cache destination offsets."""

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
            raise RuntimeError(
                "Prefix-aware prefill offload requires host KV view"
            )
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
        if self.metadata.prefix_shared_tokens is None:
            raise RuntimeError("Prefix offload requires prefix_shared_tokens")
        if not hasattr(
            self.worker_view, "async_offload_layer_kv_to_host_with_offsets"
        ):
            raise RuntimeError(
                "Prefix offload requires async_offload_layer_kv_to_host_with_offsets"
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
                    f"MLA prefill offload expects 2D or 3D KV, got {seq_key.dim()}D"
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
