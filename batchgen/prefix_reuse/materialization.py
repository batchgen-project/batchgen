"""GPU paged materialization helpers for prefix-reuse prefill."""

from __future__ import annotations

import math
from dataclasses import dataclass
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

    manager: object
    append_plan: object
    load_task: Optional[_AsyncTask] = None
    _loaded: bool = False

    def wait_for_layer(self, layer_idx: int) -> None:
        if self._loaded or self.load_task is None:
            return
        wait_for_layer = getattr(self.load_task, "wait_for_layer", None)
        if wait_for_layer is None:
            self.wait()
            return
        wait_for_layer(int(layer_idx))

    def wait(self) -> None:
        if self._loaded:
            return
        if self.load_task is not None:
            self.load_task.wait()
        self._loaded = True


@dataclass
class PrefixMaterializationBundle:
    """Materialized prefix pages keyed by logical prefix-cache group id."""

    by_group_id: dict[int, SingleGroupPrefixMaterialization]

    @classmethod
    def from_single(
        cls, group_id: int, materialization: SingleGroupPrefixMaterialization
    ) -> "PrefixMaterializationBundle":
        return cls(by_group_id={int(group_id): materialization})

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
    if int(group_id) != 0:
        raise RuntimeError(
            f"{consumer} requires prefix materialization group {group_id}, "
            "but received a legacy single-group materialization"
        )
    return materialization


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
        wait_for_layer = getattr(self._load_task, "wait_for_layer", None)
        if wait_for_layer is None:
            self.wait()
            return
        wait_for_layer(int(layer_idx))


def materialize_single_group_prefix_pages(
    *,
    gpu_manager: object,
    host_worker_view: object,
    sequences: Sequence[PrefixMaterializationSequence],
    raw_page_tokens: int | None = None,
    prefix_cache_coordinator: Optional[_PrefixCacheCoordinator] = None,
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

    page_size = int(
        raw_page_tokens
        if raw_page_tokens is not None
        else gpu_manager.config.page_size_tokens
    )
    if page_size <= 0:
        raise ValueError("raw_page_tokens must be positive")
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
        )
        active_page_counts = torch.tensor(prefix_page_counts, dtype=torch.int64)

    gpu_manager.allocate_pages_for_sequences(sequence_ids, full_lens)
    gpu_manager.rebuild_page_table(sequence_ids)
    k_ptrs, v_ptrs = gpu_manager.get_padded_3d_page_pointers()
    append_plan = gpu_manager.prepare_prefill_suffix_append(
        sequence_ids=sequence_ids,
        prefix_lens=prefix_lens,
        suffix_lens=suffix_lens,
        rebuild_page_table=False,
    )

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
