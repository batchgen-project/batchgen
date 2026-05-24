"""GPU paged materialization helpers for prefix-reuse prefill."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import torch


class _AsyncTask(Protocol):
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

    @property
    def full_tokens(self) -> int:
        return int(self.prefix_tokens) + int(self.suffix_tokens)


@dataclass
class SingleGroupPrefixMaterialization:
    """Single KV-group materialization view consumed by current adapters."""

    manager: object
    append_plan: object
    load_task: Optional[_AsyncTask] = None
    _loaded: bool = False

    def wait_for_layer(self, layer_idx: int) -> None:
        del layer_idx
        self.wait()

    def wait(self) -> None:
        if self._loaded:
            return
        if self.load_task is not None:
            self.load_task.wait()
        self._loaded = True


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
        self._attachment_handles = tuple(int(handle) for handle in attachment_handles)
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


def materialize_single_group_prefix_pages(
    *,
    gpu_manager: object,
    host_worker_view: object,
    sequences: Sequence[PrefixMaterializationSequence],
    expected_host_region_id: int = 0,
    prefix_cache_coordinator: Optional[_PrefixCacheCoordinator] = None,
) -> SingleGroupPrefixMaterialization:
    """Materialize Host prefix pages into target GPU paged KV slots.

    This helper is intentionally below the Host prefix-cache coordinator. The
    caller provides already attached/pinned Host page handles and target
    sequence ids; this function only allocates GPU pages, starts the page-id
    based Host->GPU copy, and prepares suffix append metadata.
    """

    if not sequences:
        raise ValueError("prefix materialization requires at least one sequence")

    sequence_ids = [int(item.sequence_id) for item in sequences]
    prefix_lens = [int(item.prefix_tokens) for item in sequences]
    suffix_lens = [int(item.suffix_tokens) for item in sequences]
    full_lens = [prefix + suffix for prefix, suffix in zip(prefix_lens, suffix_lens)]
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

    page_size = int(gpu_manager.config.page_size_tokens)
    prefix_page_counts = [
        int(math.ceil(prefix_len / page_size)) if prefix_len > 0 else 0
        for prefix_len in prefix_lens
    ]
    has_prefix_pages = any(count > 0 for count in prefix_page_counts)
    host_page_ids = None
    active_page_counts = None
    if has_prefix_pages:
        host_page_ids = _build_host_page_id_tensor(
            sequences,
            prefix_page_counts=prefix_page_counts,
            expected_host_region_id=expected_host_region_id,
        )
        active_page_counts = torch.tensor(prefix_page_counts, dtype=torch.int64)

    gpu_manager.allocate_pages_for_sequences(sequence_ids, full_lens)
    gpu_manager.rebuild_page_table(sequence_ids)
    k_ptrs, v_ptrs = gpu_manager.get_padded_3d_page_pointers()

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
                k_device_ptrs=k_ptrs,
                v_device_ptrs=v_ptrs,
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

    append_plan = gpu_manager.prepare_prefill_suffix_append(
        sequence_ids=sequence_ids,
        prefix_lens=prefix_lens,
        suffix_lens=suffix_lens,
        rebuild_page_table=False,
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
    expected_host_region_id: int = 0,
    prefix_cache_coordinator: Optional[_PrefixCacheCoordinator] = None,
) -> SingleGroupPrefixMaterialization:
    """Materialize a batch of C++ HostPrefixCache lookup results.

    The Host prefix-cache coordinator owns lookup, attachment lifetime, and
    eviction. This function is only the compute-path producer: it converts
    attached lookup results for one KV group into GPU paged KV materialization.
    """

    count = len(lookup_results)
    if len(sequence_ids) != count or len(prompt_lengths) != count:
        raise ValueError("lookup_results, sequence_ids, and prompt_lengths differ")

    sequences: list[PrefixMaterializationSequence] = []
    for result, sequence_id, prompt_length in zip(
        lookup_results,
        sequence_ids,
        prompt_lengths,
    ):
        prompt_len = int(prompt_length)
        cached_tokens = int(getattr(result, "common_cached_tokens"))
        if prompt_len <= 0:
            raise ValueError(
                f"prompt length must be positive for sequence {sequence_id}"
            )
        if cached_tokens < 0 or cached_tokens > prompt_len:
            raise ValueError(
                "lookup cached token count must be within prompt length for "
                f"sequence {sequence_id}: cached={cached_tokens}, "
                f"prompt={prompt_len}"
            )
        span_pages = []
        if cached_tokens > 0:
            span = _find_group_span(result, group_id=int(group_id))
            span_raw_end = int(getattr(span, "raw_end_token"))
            if span_raw_end != cached_tokens:
                raise ValueError(
                    "single-group prefix materialization requires lookup span "
                    "to match cached token boundary for sequence "
                    f"{sequence_id}: span={span_raw_end}, cached={cached_tokens}"
                )
            span_pages = list(getattr(span, "pages"))

        sequences.append(
            PrefixMaterializationSequence(
                sequence_id=int(sequence_id),
                prefix_tokens=cached_tokens,
                suffix_tokens=prompt_len - cached_tokens,
                host_pages=span_pages,
                attachment_handle=int(getattr(result, "attachment_handle", 0)),
            )
        )

    return materialize_single_group_prefix_pages(
        gpu_manager=gpu_manager,
        host_worker_view=host_worker_view,
        sequences=sequences,
        expected_host_region_id=expected_host_region_id,
        prefix_cache_coordinator=prefix_cache_coordinator,
    )


def _build_host_page_id_tensor(
    sequences: Sequence[PrefixMaterializationSequence],
    *,
    prefix_page_counts: Sequence[int],
    expected_host_region_id: int,
) -> torch.Tensor:
    max_pages = max(int(count) for count in prefix_page_counts)
    rows: list[list[int]] = []
    for item, page_count in zip(sequences, prefix_page_counts):
        pages = [
            _host_page_id(handle, expected_host_region_id=expected_host_region_id)
            for handle in item.host_pages
        ]
        if len(pages) < int(page_count):
            raise ValueError(
                "host prefix page list is shorter than required for sequence "
                f"{item.sequence_id}: need {page_count}, got {len(pages)}"
            )
        row = pages[: int(page_count)]
        row.extend([0] * (max_pages - len(row)))
        rows.append(row)
    return torch.tensor(rows, dtype=torch.int64)


def _find_group_span(result: object, *, group_id: int) -> object:
    spans = getattr(result, "materialization_spans", None)
    if spans is None:
        raise TypeError("lookup result must expose materialization_spans")
    for span in spans:
        if int(getattr(span, "group_id")) == int(group_id):
            return span
    raise ValueError(f"lookup result has no materialization span for group {group_id}")


def _host_page_id(handle: int | object, *, expected_host_region_id: int) -> int:
    if isinstance(handle, int):
        return int(handle)
    region_id = getattr(handle, "host_region_id", expected_host_region_id)
    if int(region_id) != int(expected_host_region_id):
        raise ValueError(
            "prefix materialization cannot load host page from region "
            f"{region_id}; expected region {expected_host_region_id}"
        )
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
