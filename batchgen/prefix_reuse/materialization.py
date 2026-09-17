"""Materialize attached Host prefix pages into temporary GPU paged KV."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

import torch


class _AsyncTask(Protocol):
    def wait_for_layer(self, layer_idx: int) -> None: ...

    def wait(self) -> None: ...


@dataclass
class PrefixMaterialization:
    manager: object
    append_plan: object
    load_task: _AsyncTask | None
    coordinator: object
    attachment_handles: tuple[int, ...]
    _closed: bool = False

    def wait_for_layer(self, layer_idx: int) -> None:
        if self._closed:
            raise RuntimeError("prefix materialization is already closed")
        if self.load_task is not None:
            self.load_task.wait_for_layer(int(layer_idx))

    def close(self, *, empty_cuda_cache: bool = False) -> None:
        if self._closed:
            return
        try:
            if self.load_task is not None:
                self.load_task.wait()
        finally:
            for handle in reversed(self.attachment_handles):
                self.coordinator.end_attachment_load(handle)
            self.manager.destroy(empty_cuda_cache=empty_cuda_cache)
            self._closed = True


def materialize_gpt_oss_prefixes(
    *,
    gpu_manager: object,
    host_worker_view: object,
    coordinator: object,
    lookup_results: Sequence[object],
    sequence_ids: Sequence[int],
    prompt_lengths: Sequence[int],
    compute_cached_tokens: Sequence[int],
    raw_page_tokens: int,
) -> PrefixMaterialization:
    """Build one mixed hit/miss GPT-OSS prefill materialization."""

    count = len(lookup_results)
    if not (
        len(sequence_ids)
        == len(prompt_lengths)
        == len(compute_cached_tokens)
        == count
    ):
        raise ValueError("prefix materialization inputs must have equal lengths")
    if count == 0:
        raise ValueError("prefix materialization requires at least one sequence")

    gpu_page_tokens = int(gpu_manager.config.page_size_tokens)
    host_page_tokens = int(raw_page_tokens)
    if gpu_page_tokens != host_page_tokens:
        raise RuntimeError(
            "GPT-OSS prefix materialization requires equal Host/GPU page sizes: "
            f"host={host_page_tokens}, gpu={gpu_page_tokens}"
        )

    prefix_lens = [int(value) for value in compute_cached_tokens]
    full_lens = [int(value) for value in prompt_lengths]
    suffix_lens = [
        full - prefix for full, prefix in zip(full_lens, prefix_lens)
    ]
    page_counts = [
        math.ceil(prefix / host_page_tokens) if prefix else 0
        for prefix in prefix_lens
    ]
    if any(full <= 0 for full in full_lens):
        raise ValueError("prompt lengths must be positive")
    if any(prefix < 0 or suffix <= 0 for prefix, suffix in zip(prefix_lens, suffix_lens)):
        raise ValueError(
            "computed prefix must leave at least one prompt token for prefill"
        )

    gpu_manager.allocate_pages_for_sequences(sequence_ids, full_lens)
    gpu_manager.rebuild_page_table(sequence_ids)
    append_plan = gpu_manager.prepare_prefill_suffix_append(
        sequence_ids=sequence_ids,
        prefix_lens=prefix_lens,
        suffix_lens=suffix_lens,
        rebuild_page_table=False,
    )

    max_pages = max(page_counts)
    if max_pages == 0:
        return PrefixMaterialization(
            manager=gpu_manager,
            append_plan=append_plan,
            load_task=None,
            coordinator=coordinator,
            attachment_handles=(),
        )

    host_rows: list[list[int]] = []
    attachment_handles: list[int] = []
    for result, page_count in zip(lookup_results, page_counts):
        pages: list[int] = []
        if page_count:
            spans = list(result.materialization_spans)
            if len(spans) != 1 or int(spans[0].group_id) != 0:
                raise RuntimeError(
                    "GPT-OSS prefix materialization requires FULL_KV group 0"
                )
            pages = [int(page.page_id) for page in spans[0].pages]
            if len(pages) < page_count:
                raise RuntimeError(
                    "prefix lookup returned too few Host pages for materialization"
                )
            handle = int(result.attachment_handle)
            if handle == 0:
                raise RuntimeError("prefix hit is missing its attachment handle")
            attachment_handles.append(handle)
        row = pages[:page_count]
        row.extend([0] * (max_pages - len(row)))
        host_rows.append(row)

    host_page_ids = torch.tensor(host_rows, dtype=torch.int64)
    active_page_counts = torch.tensor(page_counts, dtype=torch.int64)
    k_ptrs, v_ptrs = gpu_manager.get_padded_3d_page_pointers()

    begun: list[int] = []
    try:
        for handle in attachment_handles:
            coordinator.begin_attachment_load(handle)
            begun.append(handle)
        load_task = host_worker_view.async_load_prefix_pages_to_device(
            host_page_ids=host_page_ids,
            active_page_counts=active_page_counts,
            k_device_ptrs=k_ptrs,
            v_device_ptrs=v_ptrs,
        )
    except Exception:
        for handle in reversed(begun):
            coordinator.end_attachment_load(handle)
        gpu_manager.destroy()
        raise

    return PrefixMaterialization(
        manager=gpu_manager,
        append_plan=append_plan,
        load_task=load_task,
        coordinator=coordinator,
        attachment_handles=tuple(begun),
    )
