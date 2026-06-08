"""GPU paged materialization helpers for prefix-reuse prefill."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

import torch

from batchgen.prefix_reuse.prefill import effective_prefix_shared_tokens


class _AsyncTask(Protocol):
    def wait_for_layer(self, layer_idx: int) -> None: ...

    def wait(self) -> None: ...


class _PrefixCacheCoordinator(Protocol):
    def begin_attachment_load(self, attachment_handle: int) -> None: ...

    def end_attachment_load(self, attachment_handle: int) -> None: ...


@dataclass(frozen=True)
class PrefixMaterializationSequence:
    """Host prefix pages needed by one target GPU sequence."""

    sequence_id: int
    prefix_tokens: int
    suffix_tokens: int
    host_pages: Sequence[int | object]
    attachment_handle: int = 0


@dataclass
class SingleGroupPrefixMaterialization:
    """Single KV-group materialization view consumed by current adapters."""

    manager: object | None
    append_plan: object | None
    load_task: Optional[_AsyncTask] = None
    _loaded: bool = False
    _closed: bool = False

    def wait_for_layer(self, layer_idx: int) -> None:
        if self._closed:
            raise RuntimeError("prefix materialization is already closed")
        if self._loaded or self.load_task is None:
            return
        self.load_task.wait_for_layer(int(layer_idx))

    def wait(self) -> None:
        if self._loaded:
            return
        if self.load_task is not None:
            self.load_task.wait()
        self._loaded = True

    def close(self, *, empty_cuda_cache: bool = False) -> None:
        """Wait for outstanding loads and release GPU materialization buffers."""

        if self._closed:
            return
        manager = self.manager
        try:
            self.wait()
        finally:
            self.manager = None
            self.append_plan = None
            self.load_task = None
            self._closed = True
            if manager is not None:
                manager.destroy(empty_cuda_cache=empty_cuda_cache)

    def finish_layer(self, layer_idx: int) -> None:
        """Notify materialization that a logical layer no longer needs GPU KV."""

        del layer_idx


@dataclass
class RollingSingleGroupPrefixMaterialization(SingleGroupPrefixMaterialization):
    """Two-slot logical-layer materialization for prefix-hit prefill.

    The manager owns a small physical layer window and maps logical layers onto
    those slots. Prefix pages are loaded layer-by-layer from Host KV, so prefill
    does not retain full-model GPU KV for every layer.
    """

    host_worker_view: object | None = None
    host_page_ids: torch.Tensor | None = None
    active_page_counts: torch.Tensor | None = None
    host_page_tokens: int | None = None
    logical_layer_count: int = 0
    prefix_cache_coordinator: Optional[_PrefixCacheCoordinator] = None
    attachment_handles: Sequence[int] = ()
    _scheduled_tasks: dict[int, _AsyncTask] = field(default_factory=dict)
    _begun_handles: list[int] = field(default_factory=list)
    _attachments_released: bool = False

    def start(self) -> None:
        """Begin attachment protection and prefetch the first two layers."""

        if self.host_page_ids is None or self.active_page_counts is None:
            return
        self._begin_attachment_loads()
        try:
            self._schedule_layer(0)
            self._schedule_layer(1)
        except Exception:
            self.wait()
            raise

    def wait_for_layer(self, layer_idx: int) -> None:
        if self._closed:
            raise RuntimeError("prefix materialization is already closed")
        layer_idx = int(layer_idx)
        self._schedule_layer(layer_idx)
        task = self._scheduled_tasks.get(layer_idx)
        if task is not None:
            task.wait_for_layer(layer_idx)

    def finish_layer(self, layer_idx: int) -> None:
        if self._closed:
            return
        # Reuse the just-consumed physical slot for the next non-resident
        # logical layer. Callers invoke this after the layer's attention output
        # and suffix offload have consumed the temporary GPU KV.
        self._schedule_layer(int(layer_idx) + 2)

    def wait(self) -> None:
        if self._loaded:
            return
        try:
            for task in self._scheduled_tasks.values():
                task.wait()
        finally:
            self._release_attachment_loads()
            self._loaded = True

    def close(self, *, empty_cuda_cache: bool = False) -> None:
        if self._closed:
            return
        manager = self.manager
        try:
            self.wait()
        finally:
            self.manager = None
            self.append_plan = None
            self.load_task = None
            self.host_worker_view = None
            self.host_page_ids = None
            self.active_page_counts = None
            self._scheduled_tasks.clear()
            self._closed = True
            if manager is not None:
                manager.destroy(empty_cuda_cache=empty_cuda_cache)

    def _begin_attachment_loads(self) -> None:
        if self._begun_handles or self.prefix_cache_coordinator is None:
            return
        for handle in self.attachment_handles:
            self.prefix_cache_coordinator.begin_attachment_load(int(handle))
            self._begun_handles.append(int(handle))

    def _release_attachment_loads(self) -> None:
        if self._attachments_released:
            return
        coordinator = self.prefix_cache_coordinator
        if coordinator is not None:
            for handle in reversed(self._begun_handles):
                coordinator.end_attachment_load(int(handle))
        self._begun_handles.clear()
        self._attachments_released = True

    def _schedule_layer(self, layer_idx: int) -> None:
        if (
            layer_idx < 0
            or layer_idx >= int(self.logical_layer_count)
            or layer_idx in self._scheduled_tasks
            or self.host_page_ids is None
            or self.active_page_counts is None
        ):
            return
        if self.manager is None or self.host_worker_view is None:
            raise RuntimeError("rolling prefix materialization is not active")

        physical_layer = int(self.manager.resolve_physical_layer(layer_idx))
        selected_rows = torch.tensor([physical_layer], dtype=torch.int64)
        k_ptrs, v_ptrs = self.manager.get_padded_3d_page_pointers()
        selected_k_ptrs = k_ptrs.index_select(0, selected_rows).contiguous()
        selected_v_ptrs = (
            None
            if v_ptrs is None
            else v_ptrs.index_select(0, selected_rows).contiguous()
        )
        selected_k_ptrs, selected_v_ptrs = _expand_device_ptrs_for_host_pages(
            gpu_manager=self.manager,
            k_device_ptrs=selected_k_ptrs,
            v_device_ptrs=selected_v_ptrs,
            active_page_counts=self.active_page_counts,
            host_page_tokens=self.host_page_tokens,
        )
        logical_layers = torch.tensor([layer_idx], dtype=torch.int64)
        task = self.host_worker_view.async_load_prefix_layers_to_device(
            host_page_ids=self.host_page_ids,
            active_page_counts=self.active_page_counts,
            logical_layer_ids=logical_layers,
            k_device_ptrs=selected_k_ptrs,
            v_device_ptrs=selected_v_ptrs,
        )
        self._scheduled_tasks[layer_idx] = task


@dataclass
class PrefixMaterializationBundle:
    """Materialized prefix pages keyed by logical prefix-cache group id."""

    by_group_id: dict[int, SingleGroupPrefixMaterialization]

    def get(self, group_id: int) -> Optional[SingleGroupPrefixMaterialization]:
        return self.by_group_id.get(int(group_id))

    def require(
        self, group_id: int, *, consumer: str
    ) -> SingleGroupPrefixMaterialization:
        materialization = self.get(group_id)
        if materialization is None:
            raise RuntimeError(
                f"{consumer} requires prefix materialization group {group_id}"
            )
        return materialization

    def wait_for_layer(self, layer_idx: int) -> None:
        for materialization in self.by_group_id.values():
            materialization.wait_for_layer(layer_idx)

    def wait(self) -> None:
        for materialization in self.by_group_id.values():
            materialization.wait()

    def finish_layer(self, layer_idx: int) -> None:
        for materialization in self.by_group_id.values():
            materialization.finish_layer(layer_idx)

    def close(self, *, empty_cuda_cache: bool = False) -> None:
        for materialization in self.by_group_id.values():
            materialization.close(empty_cuda_cache=empty_cuda_cache)


def get_prefix_materialization_for_group(
    materialization: object | None,
    *,
    group_id: int,
    consumer: str,
) -> SingleGroupPrefixMaterialization | None:
    """Return the materialization consumed by one attention backend."""

    if materialization is None:
        return None
    if isinstance(materialization, PrefixMaterializationBundle):
        return materialization.require(group_id, consumer=consumer)
    raise RuntimeError(
        f"{consumer} requires PrefixMaterializationBundle, "
        f"got {type(materialization).__name__}"
    )


class _AttachmentLoadTask:
    def __init__(
        self,
        *,
        load_task: _AsyncTask,
        coordinator: _PrefixCacheCoordinator,
        attachment_handles: Sequence[int],
    ) -> None:
        self._load_task = load_task
        self._coordinator = coordinator
        self._attachment_handles = tuple(
            int(handle) for handle in attachment_handles
        )
        self._done = False

    def wait(self) -> None:
        if self._done:
            return
        try:
            self._load_task.wait()
        finally:
            for handle in reversed(self._attachment_handles):
                self._coordinator.end_attachment_load(handle)
            self._done = True

    def wait_for_layer(self, layer_idx: int) -> None:
        if self._done:
            return
        self._load_task.wait_for_layer(int(layer_idx))


def materialize_single_group_prefix_pages(
    *,
    gpu_manager: object,
    host_worker_view: object,
    sequences: Sequence[PrefixMaterializationSequence],
    raw_page_tokens: int | None = None,
    prefix_cache_coordinator: Optional[_PrefixCacheCoordinator] = None,
    rolling_logical_layer_count: int | None = None,
) -> SingleGroupPrefixMaterialization:
    """Materialize Host prefix pages into target GPU paged KV slots.

    This helper is intentionally below the Host prefix-cache coordinator. The
    caller provides already attached/pinned Host page handles and target
    sequence ids; this function only allocates GPU pages, starts the page-id
    based Host->GPU copy, and prepares suffix append metadata.
    """

    if not sequences:
        raise ValueError(
            "prefix materialization requires at least one sequence"
        )

    sequence_ids = [int(item.sequence_id) for item in sequences]
    prefix_lens = [int(item.prefix_tokens) for item in sequences]
    suffix_lens = [int(item.suffix_tokens) for item in sequences]
    full_lens = [
        prefix + suffix for prefix, suffix in zip(prefix_lens, suffix_lens)
    ]
    for seq_id, prefix_len, suffix_len, full_len in zip(
        sequence_ids, prefix_lens, suffix_lens, full_lens
    ):
        if prefix_len < 0 or suffix_len < 0:
            raise ValueError(
                "prefix/suffix lengths must be non-negative for sequence "
                f"{seq_id}: prefix={prefix_len}, suffix={suffix_len}"
            )
        if full_len <= 0:
            raise ValueError(
                f"full sequence length must be positive for sequence {seq_id}"
            )

    host_page_tokens = int(
        raw_page_tokens
        if raw_page_tokens is not None
        else gpu_manager.config.page_size_tokens
    )
    if host_page_tokens <= 0:
        raise ValueError("raw_page_tokens must be positive")
    prefix_page_counts = [
        int(math.ceil(prefix_len / host_page_tokens)) if prefix_len > 0 else 0
        for prefix_len in prefix_lens
    ]
    has_prefix_pages = any(count > 0 for count in prefix_page_counts)
    host_page_ids = None
    active_page_counts = None
    if has_prefix_pages:
        host_page_ids = _build_host_page_id_tensor(
            sequences,
            prefix_page_counts=prefix_page_counts,
        )
        active_page_counts = torch.tensor(prefix_page_counts, dtype=torch.int64)

    gpu_manager.allocate_pages_for_sequences(sequence_ids, full_lens)
    gpu_manager.rebuild_page_table(sequence_ids)
    k_ptrs, v_ptrs = gpu_manager.get_padded_3d_page_pointers()
    copy_k_ptrs, copy_v_ptrs = _expand_device_ptrs_for_host_pages(
        gpu_manager=gpu_manager,
        k_device_ptrs=k_ptrs,
        v_device_ptrs=v_ptrs,
        active_page_counts=active_page_counts,
        host_page_tokens=host_page_tokens,
    )
    append_plan = gpu_manager.prepare_prefill_suffix_append(
        sequence_ids=sequence_ids,
        prefix_lens=prefix_lens,
        suffix_lens=suffix_lens,
        rebuild_page_table=False,
    )

    if rolling_logical_layer_count is not None:
        attachment_handles = _attachment_handles_for_load(
            sequences,
            prefix_page_counts,
        )
        if attachment_handles and prefix_cache_coordinator is None:
            raise ValueError(
                "prefix materialization sequences with attachment handles "
                "require prefix_cache_coordinator"
            )
        materialization = RollingSingleGroupPrefixMaterialization(
            manager=gpu_manager,
            append_plan=append_plan,
            host_worker_view=host_worker_view,
            host_page_ids=host_page_ids,
            active_page_counts=active_page_counts,
            host_page_tokens=host_page_tokens,
            logical_layer_count=int(rolling_logical_layer_count),
            prefix_cache_coordinator=prefix_cache_coordinator,
            attachment_handles=tuple(attachment_handles),
        )
        try:
            materialization.start()
        except Exception:
            materialization.close()
            raise
        return materialization

    load_task = None
    if has_prefix_pages:
        attachment_handles = _attachment_handles_for_load(
            sequences,
            prefix_page_counts,
        )
        if attachment_handles and prefix_cache_coordinator is None:
            raise ValueError(
                "prefix materialization sequences with attachment handles "
                "require prefix_cache_coordinator"
            )

        begun_handles: list[int] = []
        try:
            if prefix_cache_coordinator is not None:
                for handle in attachment_handles:
                    prefix_cache_coordinator.begin_attachment_load(handle)
                    begun_handles.append(handle)

            load_task = host_worker_view.async_load_prefix_pages_to_device(
                host_page_ids=host_page_ids,
                active_page_counts=active_page_counts,
                k_device_ptrs=copy_k_ptrs,
                v_device_ptrs=copy_v_ptrs,
            )
        except Exception:
            if prefix_cache_coordinator is not None:
                for handle in reversed(begun_handles):
                    prefix_cache_coordinator.end_attachment_load(handle)
            raise

        if prefix_cache_coordinator is not None and begun_handles:
            load_task = _AttachmentLoadTask(
                load_task=load_task,
                coordinator=prefix_cache_coordinator,
                attachment_handles=begun_handles,
            )

    return SingleGroupPrefixMaterialization(
        manager=gpu_manager,
        append_plan=append_plan,
        load_task=load_task,
    )


def materialize_single_group_lookup_results(
    *,
    gpu_manager: object,
    host_worker_view: object,
    lookup_results: Sequence[object],
    sequence_ids: Sequence[int],
    prompt_lengths: Sequence[int],
    group_id: int,
    prefix_shared_tokens: Sequence[int] | None = None,
    raw_page_tokens: int | None = None,
    prefix_cache_coordinator: Optional[_PrefixCacheCoordinator] = None,
    rolling_logical_layer_count: int | None = None,
) -> SingleGroupPrefixMaterialization:
    """Materialize a batch of C++ HostPrefixCache lookup results.

    The Host prefix-cache coordinator owns lookup, attachment lifetime, and
    eviction. This function is only the compute-path producer: it converts
    attached lookup results for one KV group into GPU paged KV materialization.
    """

    count = len(lookup_results)
    if len(sequence_ids) != count or len(prompt_lengths) != count:
        raise ValueError(
            "lookup_results, sequence_ids, and prompt_lengths differ"
        )
    if prefix_shared_tokens is not None and len(prefix_shared_tokens) != count:
        raise ValueError(
            "prefix_shared_tokens length differs from lookup_results"
        )

    sequences: list[PrefixMaterializationSequence] = []
    for idx, (result, sequence_id, prompt_length) in enumerate(zip(
        lookup_results,
        sequence_ids,
        prompt_lengths,
    )):
        prompt_len = int(prompt_length)
        raw_cached_tokens = int(result.common_cached_tokens)
        if prompt_len <= 0:
            raise ValueError(
                f"prompt length must be positive for sequence {sequence_id}"
            )
        if raw_cached_tokens < 0 or raw_cached_tokens > prompt_len:
            raise ValueError(
                "lookup cached token count must be within prompt length for "
                f"sequence {sequence_id}: cached={raw_cached_tokens}, "
                f"prompt={prompt_len}"
            )
        if prefix_shared_tokens is None:
            cached_tokens = effective_prefix_shared_tokens(
                raw_cached_tokens=raw_cached_tokens,
                prompt_length=prompt_len,
            )
        else:
            cached_tokens = int(prefix_shared_tokens[idx])
        if cached_tokens < 0 or cached_tokens >= prompt_len:
            raise ValueError(
                "effective cached token count must be within compute bounds "
                f"for sequence {sequence_id}: cached={cached_tokens}, "
                f"prompt={prompt_len}"
            )
        span_pages = []
        attachment_handle = int(result.attachment_handle)
        if cached_tokens > 0:
            if attachment_handle == 0:
                raise ValueError(
                    "lookup result with cached prefix must have non-zero "
                    f"attachment_handle for sequence {sequence_id}"
                )
            span = _find_group_span(result, group_id=int(group_id))
            span_raw_end = int(span.raw_end_token)
            if span_raw_end < cached_tokens:
                raise ValueError(
                    "single-group prefix materialization requires lookup span "
                    "to cover the effective cached token boundary for sequence "
                    f"{sequence_id}: span={span_raw_end}, "
                    f"effective_cached={cached_tokens}"
                )
            span_pages = list(span.pages)

        sequences.append(
            PrefixMaterializationSequence(
                sequence_id=int(sequence_id),
                prefix_tokens=cached_tokens,
                suffix_tokens=prompt_len - cached_tokens,
                host_pages=span_pages,
                attachment_handle=attachment_handle,
            )
        )

    return materialize_single_group_prefix_pages(
        gpu_manager=gpu_manager,
        host_worker_view=host_worker_view,
        sequences=sequences,
        raw_page_tokens=raw_page_tokens,
        prefix_cache_coordinator=prefix_cache_coordinator,
        rolling_logical_layer_count=rolling_logical_layer_count,
    )


def _build_host_page_id_tensor(
    sequences: Sequence[PrefixMaterializationSequence],
    *,
    prefix_page_counts: Sequence[int],
) -> torch.Tensor:
    max_pages = max(int(count) for count in prefix_page_counts)
    rows: list[list[int]] = []
    for item, page_count in zip(sequences, prefix_page_counts):
        pages = [_host_page_id(handle) for handle in item.host_pages]
        if len(pages) < int(page_count):
            raise ValueError(
                "host prefix page list is shorter than required for sequence "
                f"{item.sequence_id}: need {page_count}, got {len(pages)}"
            )
        row = pages[: int(page_count)]
        row.extend([0] * (max_pages - len(row)))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.int64)


def _expand_device_ptrs_for_host_pages(
    *,
    gpu_manager: object,
    k_device_ptrs: torch.Tensor,
    v_device_ptrs: torch.Tensor | None,
    active_page_counts: torch.Tensor | None,
    host_page_tokens: int | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Map host-page copy slots onto potentially larger GPU pages.

    Host prefix pages are indexed by the Host KV group's raw page size. Some
    GPU kernels impose a larger paged-cache block size; for example FA3 paged
    ``flash_attn_with_kvcache`` requires a 256-token GPU page. In that case
    each Host page is copied into a subrange of the larger GPU page by adding a
    byte offset to the destination page pointer. The C++ copy path remains
    asynchronous and still copies one Host page per entry.
    """

    gpu_page_tokens = int(gpu_manager.config.page_size_tokens)
    host_tokens = int(
        host_page_tokens if host_page_tokens is not None else gpu_page_tokens
    )
    if (
        host_tokens == gpu_page_tokens
        or host_tokens > gpu_page_tokens
        or active_page_counts is None
    ):
        return k_device_ptrs, v_device_ptrs
    if host_tokens <= 0 or gpu_page_tokens <= 0:
        raise ValueError("host and GPU page sizes must be positive")
    if gpu_page_tokens % host_tokens != 0:
        raise ValueError(
            "GPU page size must be a multiple of Host page size for prefix "
            f"materialization, got gpu={gpu_page_tokens}, host={host_tokens}"
        )

    k_page_bytes = _host_page_bytes(
        gpu_manager=gpu_manager,
        host_page_tokens=host_tokens,
        is_value=False,
    )
    expanded_k = _expand_pointer_tensor_for_host_pages(
        k_device_ptrs,
        active_page_counts=active_page_counts,
        host_page_bytes=k_page_bytes,
        host_pages_per_gpu_page=gpu_page_tokens // host_tokens,
    )

    expanded_v = None
    if v_device_ptrs is not None:
        v_page_bytes = _host_page_bytes(
            gpu_manager=gpu_manager,
            host_page_tokens=host_tokens,
            is_value=True,
        )
        expanded_v = _expand_pointer_tensor_for_host_pages(
            v_device_ptrs,
            active_page_counts=active_page_counts,
            host_page_bytes=v_page_bytes,
            host_pages_per_gpu_page=gpu_page_tokens // host_tokens,
        )

    return expanded_k, expanded_v


def _host_page_bytes(
    *,
    gpu_manager: object,
    host_page_tokens: int,
    is_value: bool,
) -> int:
    config = gpu_manager.config
    if is_value:
        heads = int(config.num_v_heads)
        head_dim = int(config.v_head_dim)
    else:
        heads = int(config.num_k_heads)
        head_dim = int(config.k_head_dim)
    element_size = torch.empty((), dtype=config.kv_dtype).element_size()
    return int(host_page_tokens) * heads * head_dim * int(element_size)


def _expand_pointer_tensor_for_host_pages(
    pointer_tensor: torch.Tensor,
    *,
    active_page_counts: torch.Tensor,
    host_page_bytes: int,
    host_pages_per_gpu_page: int,
) -> torch.Tensor:
    if int(active_page_counts.numel()) == 0:
        return pointer_tensor[:, :, :0].contiguous()
    max_host_pages = int(active_page_counts.max().item())
    if max_host_pages == 0:
        return pointer_tensor[:, :, :0].contiguous()

    host_slots = torch.arange(max_host_pages, dtype=torch.long)
    gpu_slots = torch.div(
        host_slots,
        int(host_pages_per_gpu_page),
        rounding_mode="floor",
    )
    if int(gpu_slots[-1].item()) >= int(pointer_tensor.shape[2]):
        raise ValueError(
            "GPU page pointer tensor is too small for Host prefix pages: "
            f"max_host_pages={max_host_pages}, "
            f"host_pages_per_gpu_page={host_pages_per_gpu_page}, "
            f"gpu_pointer_pages={pointer_tensor.shape[2]}"
        )
    offsets = (
        torch.remainder(host_slots, int(host_pages_per_gpu_page)).to(
            dtype=torch.int64
        )
        * int(host_page_bytes)
    )
    expanded = pointer_tensor.index_select(2, gpu_slots).contiguous()
    return expanded + offsets.view(1, 1, -1)


def _find_group_span(result: object, *, group_id: int) -> object:
    spans = result.materialization_spans
    if spans is None:
        raise TypeError("lookup result must expose materialization_spans")
    for span in spans:
        if int(span.group_id) == int(group_id):
            return span
    raise ValueError(
        f"lookup result has no materialization span for group {group_id}"
    )


def _host_page_id(handle: int | object) -> int:
    if isinstance(handle, int):
        return int(handle)
    page_id = getattr(handle, "page_id", None)
    if page_id is None:
        raise TypeError("host page handle must be an int or expose page_id")
    return int(page_id)


def _attachment_handles_for_load(
    sequences: Sequence[PrefixMaterializationSequence],
    prefix_page_counts: Sequence[int],
) -> list[int]:
    handles: list[int] = []
    seen: set[int] = set()
    for item, page_count in zip(sequences, prefix_page_counts):
        handle = int(item.attachment_handle)
        if int(page_count) <= 0 or handle == 0 or handle in seen:
            continue
        seen.add(handle)
        handles.append(handle)
    return handles
